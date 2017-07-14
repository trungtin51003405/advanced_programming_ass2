	/******************************************************************************************
	                                         NEURON
	*******************************************************************************************/

	function Neuron() {
	  this.ID = Neuron.uid();

	  this.connections = {
	    inputs: {},
	    projected: {},
	    gated: {}
	  };
	  this.error = {
	    responsibility: 0,
	    projected: 0,
	    gated: 0
	  };
	  this.trace = {
	    elegibility: {},
	    extended: {},
	    influences: {}
	  };
	  this.state = 0;
	  this.old = 0;
	  this.activation = 0;
	  this.selfconnection = new Neuron.connection(this, this, 0); // weight = 0 -> not connected
	  this.transfer = Neuron.transfer.LOGISTIC;
	  this.neighboors = {};
	  this.bias = Math.random() * .2 - .1;
	}

	Neuron.prototype = {

	  // activate the neuron
	  activate: function(input) {
	    // activation from enviroment (for input neurons)
	    if (typeof input != 'undefined') {
	      this.activation = input;
	      this.derivative = 0;
	      this.bias = 0;
	      return this.activation;
	    }

	    // old state
	    this.old = this.state;

	    // eq. 15
	    this.state = this.selfconnection.gain * this.selfconnection.weight *
	      this.state + this.bias;

	    for (var i in this.connections.inputs) {
	      var input = this.connections.inputs[i];
	      this.state += input.from.activation * input.weight * input.gain;
	    }

	    // eq. 16
	    this.activation = this.transfer(this.state);

	    // f'(s)
	    this.derivative = this.transfer(this.state, true);

	    // update traces
	    var influences = [];
	    for (var id in this.trace.extended) {
	      // extended elegibility trace
	      var neuron = this.neighboors[id];

	      // if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state
	      var influence = neuron.selfconnection.gater == this ? neuron.old : 0;

	      // index runs over all the incoming connections to the gated neuron that are gated by this unit
	      for (var incoming in this.trace.influences[neuron.ID]) { // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit
	        influence += this.trace.influences[neuron.ID][incoming].weight *
	          this.trace.influences[neuron.ID][incoming].from.activation;
	      }
	      influences[neuron.ID] = influence;
	    }

	    for (var i in this.connections.inputs) {
	      var input = this.connections.inputs[i];

	      // elegibility trace - Eq. 17
	      this.trace.elegibility[input.ID] = this.selfconnection.gain * this.selfconnection
	        .weight * this.trace.elegibility[input.ID] + input.gain * input.from
	        .activation;

	      for (var id in this.trace.extended) {
	        // extended elegibility trace
	        var xtrace = this.trace.extended[id];
	        var neuron = this.neighboors[id];
	        var influence = influences[neuron.ID];

	        // eq. 18
	        xtrace[input.ID] = neuron.selfconnection.gain * neuron.selfconnection
	          .weight * xtrace[input.ID] + this.derivative * this.trace.elegibility[
	            input.ID] * influence;
	      }
	    }

	    //  update gated connection's gains
	    for (var connection in this.connections.gated) {
	      this.connections.gated[connection].gain = this.activation;
	    }

	    return this.activation;
	  },

	  // back-propagate the error
	  propagate: function(rate, target) {
	    // error accumulator
	    var error = 0;

	    // whether or not this neuron is in the output layer
	    var isOutput = typeof target != 'undefined';

	    // output neurons get their error from the enviroment
	    if (isOutput)
	      this.error.responsibility = this.error.projected = target - this.activation; // Eq. 10

	    else // the rest of the neuron compute their error responsibilities by backpropagation
	    {
	      // error responsibilities from all the connections projected from this neuron
	      for (var id in this.connections.projected) {
	        var connection = this.connections.projected[id];
	        var neuron = connection.to;
	        // Eq. 21
	        error += neuron.error.responsibility * connection.gain * connection.weight;
	      }

	      // projected error responsibility
	      this.error.projected = this.derivative * error;

	      error = 0;
	      // error responsibilities from all the connections gated by this neuron
	      for (var id in this.trace.extended) {
	        var neuron = this.neighboors[id]; // gated neuron
	        var influence = neuron.selfconnection.gater == this ? neuron.old : 0; // if gated neuron's selfconnection is gated by this neuron

	        // index runs over all the connections to the gated neuron that are gated by this neuron
	        for (var input in this.trace.influences[id]) { // captures the effect that the input connection of this neuron have, on a neuron which its input/s is/are gated by this neuron
	          influence += this.trace.influences[id][input].weight * this.trace.influences[
	            neuron.ID][input].from.activation;
	        }
	        // eq. 22
	        error += neuron.error.responsibility * influence;
	      }

	      // gated error responsibility
	      this.error.gated = this.derivative * error;

	      // error responsibility - Eq. 23
	      this.error.responsibility = this.error.projected + this.error.gated;
	    }

	    // learning rate
	    rate = rate || .1;

	    // adjust all the neuron's incoming connections
	    for (var id in this.connections.inputs) {
	      var input = this.connections.inputs[id];

	      // Eq. 24
	      var gradient = this.error.projected * this.trace.elegibility[input.ID];
	      for (var id in this.trace.extended) {
	        var neuron = this.neighboors[id];
	        gradient += neuron.error.responsibility * this.trace.extended[
	          neuron.ID][input.ID];
	      }
	      input.weight += rate * gradient; // adjust weights - aka learn
	    }

	    // adjust bias
	    this.bias += rate * this.error.responsibility;
	  },

	  project: function(neuron, weight) {
	    // self-connection
	    if (neuron == this) {
	      this.selfconnection.weight = 1;
	      return this.selfconnection;
	    }

	    // check if connection already exists
	    var connected = this.connected(neuron);
	    if (connected && connected.type == "projected") {
	      // update connection
	      if (typeof weight != 'undefined')
	        connected.connection.weight = weight;
	      // return existing connection
	      return connected.connection;
	    } else {
	      // create a new connection
	      var connection = new Neuron.connection(this, neuron, weight);
	    }

	    // reference all the connections and traces
	    this.connections.projected[connection.ID] = connection;
	    this.neighboors[neuron.ID] = neuron;
	    neuron.connections.inputs[connection.ID] = connection;
	    neuron.trace.elegibility[connection.ID] = 0;

	    for (var id in neuron.trace.extended) {
	      var trace = neuron.trace.extended[id];
	      trace[connection.ID] = 0;
	    }

	    return connection;
	  },

	  // returns true or false whether the neuron is connected to another neuron (parameter)
	  connected: function(neuron) {
	    var result = {
	      type: null,
	      connection: false
	    };

	    if (this == neuron) {
	      if (this.selfconnected()) {
	        result.type = 'selfconnection';
	        result.connection = this.selfconnection;
	        return result;
	      } else
	        return false;
	    }

	    for (var type in this.connections) {
	      for (var connection in this.connections[type]) {
	        var connection = this.connections[type][connection];
	        if (connection.to == neuron) {
	          result.type = type;
	          result.connection = connection;
	          return result;
	        } else if (connection.from == neuron) {
	          result.type = type;
	          result.connection = connection;
	          return result;
	        }
	      }
	    }

	    return false;
	  },

	  // clears all the traces (the neuron forgets it's context, but the connections remain intact)
	  clear: function() {
	    for (var trace in this.trace.elegibility){
	      this.trace.elegibility[trace] = 0;
	    }

	    for (var trace in this.trace.extended){
	      for (var extended in this.trace.extended[trace]){
	        this.trace.extended[trace][extended] = 0;
	      }
	    }

	    this.error.responsibility = this.error.projected = this.error.gated = 0;
	  },

	  // all the connections are randomized and the traces are cleared
	  reset: function() {
	    this.clear();

	    for (var type in this.connections){
	      for (var connection in this.connections[type]){
	        this.connections[type][connection].weight = Math.random() * .2 - .1;
	      }
	    }

	    this.bias = Math.random() * .2 - .1;
	    this.old = this.state = this.activation = 0;
	  },

	  // hardcodes the behaviour of the neuron into an optimized function
	  optimize: function(optimized, layer) {

	    optimized = optimized || {};
	    var store_activation = [];
	    var store_trace = [];
	    var store_propagation = [];
	    var varID = optimized.memory || 0;
	    var neurons = optimized.neurons || 1;
	    var inputs = optimized.inputs || [];
	    var targets = optimized.targets || [];
	    var outputs = optimized.outputs || [];
	    var variables = optimized.variables || {};
	    var activation_sentences = optimized.activation_sentences || [];
	    var trace_sentences = optimized.trace_sentences || [];
	    var propagation_sentences = optimized.propagation_sentences || [];
	    var layers = optimized.layers || { __count: 0, __neuron: 0 };

	    // allocate sentences
	    var allocate = function(store){
	      var allocated = layer in layers && store[layers.__count];
	      if (!allocated)
	      {
	        layers.__count = store.push([]) - 1;
	        layers[layer] = layers.__count;
	      }
	    };
	    allocate(activation_sentences);
	    allocate(trace_sentences);
	    allocate(propagation_sentences);
	    var currentLayer = layers.__count;

	    // get/reserve space in memory by creating a unique ID for a variablel
	    var getVar = function() {
	      var args = Array.prototype.slice.call(arguments);

	      if (args.length == 1) {
	        if (args[0] == 'target') {
	          var id = 'target_' + targets.length;
	          targets.push(varID);
	        } else
	          var id = args[0];
	        if (id in variables)
	          return variables[id];
	        return variables[id] = {
	          value: 0,
	          id: varID++
	        };
	      } else {
	        var extended = args.length > 2;
	        if (extended)
	          var value = args.pop();

	        var unit = args.shift();
	        var prop = args.pop();

	        if (!extended)
	          var value = unit[prop];

	        var id = prop + '_';
	        for (var i = 0; i < args.length; i++)
	          id += args[i] + '_';
	        id += unit.ID;
	        if (id in variables)
	          return variables[id];

	        return variables[id] = {
	          value: value,
	          id: varID++
	        };
	      }
	    };

	    // build sentence
	    var buildSentence = function() {
	      var args = Array.prototype.slice.call(arguments);
	      var store = args.pop();
	      var sentence = "";
	      for (var i = 0; i < args.length; i++)
	        if (typeof args[i] == 'string')
	          sentence += args[i];
	        else
	          sentence += 'F[' + args[i].id + ']';

	      store.push(sentence + ';');
	    };

	    // helper to check if an object is empty
	    var isEmpty = function(obj) {
	      for (var prop in obj) {
	        if (obj.hasOwnProperty(prop))
	          return false;
	      }
	      return true;
	    };

	    // characteristics of the neuron
	    var noProjections = isEmpty(this.connections.projected);
	    var noGates = isEmpty(this.connections.gated);
	    var isInput = layer == 'input' ? true : isEmpty(this.connections.inputs);
	    var isOutput = layer == 'output' ? true : noProjections && noGates;

	    // optimize neuron's behaviour
	    var rate = getVar('rate');
	    var activation = getVar(this, 'activation');
	    if (isInput)
	      inputs.push(activation.id);
	    else {
	      activation_sentences[currentLayer].push(store_activation);
	      trace_sentences[currentLayer].push(store_trace);
	      propagation_sentences[currentLayer].push(store_propagation);
	      var old = getVar(this, 'old');
	      var state = getVar(this, 'state');
	      var bias = getVar(this, 'bias');

		  
	      buildSentence(old, ' = ', state, store_activation);
	      buildSentence(state, ' = ', bias, store_activation);
	      for (var i in this.connections.inputs) {
	        var input = this.connections.inputs[i];
	        var input_activation = getVar(input.from, 'activation');
	        var input_weight = getVar(input, 'weight');
	        if (input.gater)
	          var input_gain = getVar(input, 'gain');
	        if (this.connections.inputs[i].gater)
	          buildSentence(state, ' += ', input_activation, ' * ',
	            input_weight, ' * ', input_gain, store_activation);
	        else
	          buildSentence(state, ' += ', input_activation, ' * ',
	            input_weight, store_activation);
	      }
	      var derivative = getVar(this, 'derivative');
	      switch (this.transfer) {
	        case Neuron.transfer.LOGISTIC:
	          buildSentence(activation, ' = (1 / (1 + Math.exp(-', state, ')))',
	            store_activation);
	          buildSentence(derivative, ' = ', activation, ' * (1 - ',
	            activation, ')', store_activation);
	          break;
	        case Neuron.transfer.TANH:
	          var eP = getVar('aux');
	          var eN = getVar('aux_2');
	          buildSentence(eP, ' = Math.exp(', state, ')', store_activation);
	          buildSentence(eN, ' = 1 / ', eP, store_activation);
	          buildSentence(activation, ' = (', eP, ' - ', eN, ') / (', eP, ' + ', eN, ')', store_activation);
	          buildSentence(derivative, ' = 1 - (', activation, ' * ', activation, ')', store_activation);
	          break;
	        case Neuron.transfer.IDENTITY:
	          buildSentence(activation, ' = ', state, store_activation);
	          buildSentence(derivative, ' = 1', store_activation);
	          break;
	        case Neuron.transfer.RELU:
	          buildSentence(activation, ' = ', state, ' > 0 ? ', state, ' : 0', store_activation);
	          buildSentence(derivative, ' = ', state, ' > 0 ? 1 : 0', store_activation);
	          break;
	      }

	      for (var id in this.trace.extended) {
	        // calculate extended elegibility traces in advance
	        var neuron = this.neighboors[id];
	        var influence = getVar('influences[' + neuron.ID + ']');
	        var neuron_old = getVar(neuron, 'old');
	        var initialized = false;
	        if (neuron.selfconnection.gater == this)
	        {
	          buildSentence(influence, ' = ', neuron_old, store_trace);
	          initialized = true;
	        }
	        for (var incoming in this.trace.influences[neuron.ID]) {
	          var incoming_weight = getVar(this.trace.influences[neuron.ID]
	            [incoming], 'weight');
	          var incoming_activation = getVar(this.trace.influences[neuron.ID]
	            [incoming].from, 'activation');

	          if (initialized)
	            buildSentence(influence, ' += ', incoming_weight, ' * ', incoming_activation, store_trace);
	          else {
	            buildSentence(influence, ' = ', incoming_weight, ' * ', incoming_activation, store_trace);
	            initialized = true;
	          }
	        }
	      }

	      for (var i in this.connections.inputs) {
	        var input = this.connections.inputs[i];
	        if (input.gater)
	          var input_gain = getVar(input, 'gain');
	        var input_activation = getVar(input.from, 'activation');
	        var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
	          .elegibility[input.ID]);
	        
	        if (input.gater)
	          buildSentence(trace, ' = ', input_gain, ' * ', input_activation,
	            store_trace);
	        else
	          buildSentence(trace, ' = ', input_activation, store_trace);
	        
	        for (var id in this.trace.extended) {
	          // extended elegibility trace
	          var neuron = this.neighboors[id];
	          var influence = getVar('influences[' + neuron.ID + ']');

	          var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
	            .elegibility[input.ID]);
	          var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID,
	            this.trace.extended[neuron.ID][input.ID]);
	          if (neuron.selfconnected())
	            var neuron_self_weight = getVar(neuron.selfconnection, 'weight');
	          if (neuron.selfconnection.gater)
	            var neuron_self_gain = getVar(neuron.selfconnection, 'gain');
	          if (neuron.selfconnected())
	            if (neuron.selfconnection.gater)
	              buildSentence(xtrace, ' = ', neuron_self_gain, ' * ',
	                neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ',
	                trace, ' * ', influence, store_trace);
	            else
	              buildSentence(xtrace, ' = ', neuron_self_weight, ' * ',
	                xtrace, ' + ', derivative, ' * ', trace, ' * ',
	                influence, store_trace);
	          else
	            buildSentence(xtrace, ' = ', derivative, ' * ', trace, ' * ',
	              influence, store_trace);
	        }
	      }
	      for (var connection in this.connections.gated) {
	        var gated_gain = getVar(this.connections.gated[connection], 'gain');
	        buildSentence(gated_gain, ' = ', activation, store_activation);
	      }
	    }
	    if (!isInput) {
	      var responsibility = getVar(this, 'error', 'responsibility', this.error
	        .responsibility);
	      if (isOutput) {
	        var target = getVar('target');
	        buildSentence(responsibility, ' = ', target, ' - ', activation,
	          store_propagation);
	        for (var id in this.connections.inputs) {
	          var input = this.connections.inputs[id];
	          var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
	            .elegibility[input.ID]);
	          var input_weight = getVar(input, 'weight');
	          buildSentence(input_weight, ' += ', rate, ' * (', responsibility,
	            ' * ', trace, ')', store_propagation);
	        }
	        outputs.push(activation.id);
	      } else {
	        if (!noProjections && !noGates) {
	          var error = getVar('aux');
	          for (var id in this.connections.projected) {
	            var connection = this.connections.projected[id];
	            var neuron = connection.to;
	            var connection_weight = getVar(connection, 'weight');
	            var neuron_responsibility = getVar(neuron, 'error',
	              'responsibility', neuron.error.responsibility);
	            if (connection.gater) {
	              var connection_gain = getVar(connection, 'gain');
	              buildSentence(error, ' += ', neuron_responsibility, ' * ',
	                connection_gain, ' * ', connection_weight,
	                store_propagation);
	            } else
	              buildSentence(error, ' += ', neuron_responsibility, ' * ',
	                connection_weight, store_propagation);
	          }
	          var projected = getVar(this, 'error', 'projected', this.error.projected);
	          buildSentence(projected, ' = ', derivative, ' * ', error,
	            store_propagation);
	          buildSentence(error, ' = 0', store_propagation);
	          for (var id in this.trace.extended) {
	            var neuron = this.neighboors[id];
	            var influence = getVar('aux_2');
	            var neuron_old = getVar(neuron, 'old');
	            if (neuron.selfconnection.gater == this)
	              buildSentence(influence, ' = ', neuron_old, store_propagation);
	            else
	              buildSentence(influence, ' = 0', store_propagation);
	            for (var input in this.trace.influences[neuron.ID]) {
	              var connection = this.trace.influences[neuron.ID][input];
	              var connection_weight = getVar(connection, 'weight');
	              var neuron_activation = getVar(connection.from, 'activation');
	              buildSentence(influence, ' += ', connection_weight, ' * ',
	                neuron_activation, store_propagation);
	            }
	            var neuron_responsibility = getVar(neuron, 'error',
	              'responsibility', neuron.error.responsibility);
	            buildSentence(error, ' += ', neuron_responsibility, ' * ',
	              influence, store_propagation);
	          }
	          var gated = getVar(this, 'error', 'gated', this.error.gated);
	          buildSentence(gated, ' = ', derivative, ' * ', error,
	            store_propagation);
	          buildSentence(responsibility, ' = ', projected, ' + ', gated,
	            store_propagation);
	          for (var id in this.connections.inputs) {
	            var input = this.connections.inputs[id];
	            var gradient = getVar('aux');
	            var trace = getVar(this, 'trace', 'elegibility', input.ID, this
	              .trace.elegibility[input.ID]);
	            buildSentence(gradient, ' = ', projected, ' * ', trace,
	              store_propagation);
	            for (var id in this.trace.extended) {
	              var neuron = this.neighboors[id];
	              var neuron_responsibility = getVar(neuron, 'error',
	                'responsibility', neuron.error.responsibility);
	              var xtrace = getVar(this, 'trace', 'extended', neuron.ID,
	                input.ID, this.trace.extended[neuron.ID][input.ID]);
	              buildSentence(gradient, ' += ', neuron_responsibility, ' * ',
	                xtrace, store_propagation);
	            }
	            var input_weight = getVar(input, 'weight');
	            buildSentence(input_weight, ' += ', rate, ' * ', gradient,
	              store_propagation);
	          }

	        } else if (noGates) {
	          buildSentence(responsibility, ' = 0', store_propagation);
	          for (var id in this.connections.projected) {
	            var connection = this.connections.projected[id];
	            var neuron = connection.to;
	            var connection_weight = getVar(connection, 'weight');
	            var neuron_responsibility = getVar(neuron, 'error',
	              'responsibility', neuron.error.responsibility);
	            if (connection.gater) {
	              var connection_gain = getVar(connection, 'gain');
	              buildSentence(responsibility, ' += ', neuron_responsibility,
	                ' * ', connection_gain, ' * ', connection_weight,
	                store_propagation);
	            } else
	              buildSentence(responsibility, ' += ', neuron_responsibility,
	                ' * ', connection_weight, store_propagation);
	          }
	          buildSentence(responsibility, ' *= ', derivative,
	            store_propagation);
	          for (var id in this.connections.inputs) {
	            var input = this.connections.inputs[id];
	            var trace = getVar(this, 'trace', 'elegibility', input.ID, this
	              .trace.elegibility[input.ID]);
	            var input_weight = getVar(input, 'weight');
	            buildSentence(input_weight, ' += ', rate, ' * (',
	              responsibility, ' * ', trace, ')', store_propagation);
	          }
	        } else if (noProjections) {
	          buildSentence(responsibility, ' = 0', store_propagation);
	          for (var id in this.trace.extended) {
	            var neuron = this.neighboors[id];
	            var influence = getVar('aux');
	            var neuron_old = getVar(neuron, 'old');
	            if (neuron.selfconnection.gater == this)
	              buildSentence(influence, ' = ', neuron_old, store_propagation);
	            else
	              buildSentence(influence, ' = 0', store_propagation);
	            for (var input in this.trace.influences[neuron.ID]) {
	              var connection = this.trace.influences[neuron.ID][input];
	              var connection_weight = getVar(connection, 'weight');
	              var neuron_activation = getVar(connection.from, 'activation');
	              buildSentence(influence, ' += ', connection_weight, ' * ',
	                neuron_activation, store_propagation);
	            }
	            var neuron_responsibility = getVar(neuron, 'error',
	              'responsibility', neuron.error.responsibility);
	            buildSentence(responsibility, ' += ', neuron_responsibility,
	              ' * ', influence, store_propagation);
	          }
	          buildSentence(responsibility, ' *= ', derivative,
	            store_propagation);
	          for (var id in this.connections.inputs) {
	            var input = this.connections.inputs[id];
	            var gradient = getVar('aux');
	            buildSentence(gradient, ' = 0', store_propagation);
	            for (var id in this.trace.extended) {
	              var neuron = this.neighboors[id];
	              var neuron_responsibility = getVar(neuron, 'error',
	                'responsibility', neuron.error.responsibility);
	              var xtrace = getVar(this, 'trace', 'extended', neuron.ID,
	                input.ID, this.trace.extended[neuron.ID][input.ID]);
	              buildSentence(gradient, ' += ', neuron_responsibility, ' * ',
	                xtrace, store_propagation);
	            }
	            var input_weight = getVar(input, 'weight');
	            buildSentence(input_weight, ' += ', rate, ' * ', gradient,
	              store_propagation);
	          }
	        }
	      }
	      buildSentence(bias, ' += ', rate, ' * ', responsibility,
	        store_propagation);
	    }
	    return {
	      memory: varID,
	      neurons: neurons + 1,
	      inputs: inputs,
	      outputs: outputs,
	      targets: targets,
	      variables: variables,
	      activation_sentences: activation_sentences,
	      trace_sentences: trace_sentences,
	      propagation_sentences: propagation_sentences,
	      layers: layers
	    }
	  }
	}

	// represents a connection between two neurons
	Neuron.connection = function Connection(from, to, weight) {
	  if (!from || !to)
	    throw new Error("Connection Error: Invalid neurons");

	  this.ID = Neuron.connection.uid();
	  this.from = from;
	  this.to = to;
	  this.weight = typeof weight == 'undefined' ? Math.random() * .2 - .1 : weight;
	  this.gain = 1;
	  this.gater = null;
	}

	// squashing functions
	Neuron.transfer = {};

	// eq. 5 & 5'
	Neuron.transfer.LOGISTIC = function(x, derivate) {
	  var fx = 1 / (1 + Math.exp(-x));
	  if (!derivate)
	    return fx;
	  return fx * (1 - fx);
	};
	Neuron.transfer.TANH = function(x, derivate) {
	  if(derivate)
	    return 1 - Math.pow(Math.tanh(x), 2);
	  return Math.tanh(x);
	};
	Neuron.transfer.IDENTITY = function(x, derivate) {
	  return derivate ? 1 : x;
	};
	Neuron.transfer.HLIM = function(x, derivate) {
	  return derivate ? 1 : x > 0 ? 1 : 0;
	};
	Neuron.transfer.RELU = function(x, derivate) {
	  if (derivate)
	    return x > 0 ? 1 : 0;
	  return x > 0 ? x : 0;
	};

	// unique ID's
	(function() {
	  var neurons = 0;
	  var connections = 0;
	  Neuron.uid = function() {
	    return neurons++;
	  }
	  Neuron.connection.uid = function() {
	    return connections++;
	  }
	  Neuron.quantity = function() {
	    return {
	      neurons: neurons,
	      connections: connections
	    }
	  }
	})();



	/*******************************************************************************************
	                                            LAYER
	*******************************************************************************************/

	function Layer(size) {
	  this.size = size | 0;
	  this.list = [];

	  this.connectedTo = [];

	  while (size--) {
	    var neuron = new Neuron();
	    this.list.push(neuron);
	  }
	}

	Layer.prototype = {

	  // activates all the neurons in the layer
	  activate: function(input) {

	    var activations = [];

	    if (typeof input != 'undefined') {
	      if (input.length != this.size)
	        throw new Error("INPUT size and LAYER size must be the same to activate!");

	      for (var id in this.list) {
	        var neuron = this.list[id];
	        var activation = neuron.activate(input[id]);
	        activations.push(activation);
	      }
	    } else {
	      for (var id in this.list) {
	        var neuron = this.list[id];
	        var activation = neuron.activate();
	        activations.push(activation);
	      }
	    }
	    return activations;
	  },

	  // propagates the error on all the neurons of the layer
	  propagate: function(rate, target) {

	    if (typeof target != 'undefined') {
	      if (target.length != this.size)
	        throw new Error("TARGET size and LAYER size must be the same to propagate!");

	      for (var id = this.list.length - 1; id >= 0; id--) {
	        var neuron = this.list[id];
	        neuron.propagate(rate, target[id]);
	      }
	    } else {
	      for (var id = this.list.length - 1; id >= 0; id--) {
	        var neuron = this.list[id];
	        neuron.propagate(rate);
	      }
	    }
	  },

	  // projects a connection from this layer to another one
	  project: function(layer, type, weights) {

	    if (layer instanceof Network)
	      layer = layer.layers.input;

	    if (layer instanceof Layer) {
	      if (!this.connected(layer))
	        return new Layer.connection(this, layer, type, weights);
	    } else
	      throw new Error("Invalid argument, you can only project connections to LAYERS and NETWORKS!");


	  },

	  // true or false whether the whole layer is self-connected or not
	  selfconnected: function() {

	    for (var id in this.list) {
	      var neuron = this.list[id];
	      if (!neuron.selfconnected())
	        return false;
	    }
	    return true;
	  },

	  // true of false whether the layer is connected to another layer (parameter) or not
	  connected: function(layer) {
	    // Check if ALL to ALL connection
	    var connections = 0;
	    for (var here in this.list) {
	      for (var there in layer.list) {
	        var from = this.list[here];
	        var to = layer.list[there];
	        var connected = from.connected(to);
	        if (connected.type == 'projected')
	          connections++;
	      }
	    }
	    if (connections == this.size * layer.size)
	      return Layer.connectionType.ALL_TO_ALL;

	    // Check if ONE to ONE connection
	    connections = 0;
	    for (var neuron in this.list) {
	      var from = this.list[neuron];
	      var to = layer.list[neuron];
	      var connected = from.connected(to);
	      if (connected.type == 'projected')
	        connections++;
	    }
	    if (connections == this.size)
	      return Layer.connectionType.ONE_TO_ONE;
	  },

	  // clears all the neuorns in the layer
	  clear: function() {
	    for (var id in this.list) {
	      var neuron = this.list[id];
	      neuron.clear();
	    }
	  },

	  // resets all the neurons in the layer
	  reset: function() {
	    for (var id in this.list) {
	      var neuron = this.list[id];
	      neuron.reset();
	    }
	  },

	  // returns all the neurons in the layer (array)
	  neurons: function() {
	    return this.list;
	  },

	  // adds a neuron to the layer
	  add: function(neuron) {
	    this.neurons[neuron.ID] = neuron || new Neuron();
	    this.list.push(neuron);
	    this.size++;
	  },

	  set: function(options) {
	    options = options || {};

	    for (var i in this.list) {
	      var neuron = this.list[i];
	      if (options.label)
	        neuron.label = options.label + '_' + neuron.ID;
	      if (options.transfer)
	        neuron.transfer = options.transfer;
	      if (options.bias)
	        neuron.bias = options.bias;
	    }
	    return this;
	  }
	}

	// represents a connection from one layer to another, and keeps track of its weight and gain
	Layer.connection = function LayerConnection(fromLayer, toLayer, type, weights) {
	  this.ID = Layer.connection.uid();
	  this.from = fromLayer;
	  this.to = toLayer;
	  this.selfconnection = toLayer == fromLayer;
	  this.type = type;
	  this.connections = {};
	  this.list = [];
	  this.size = 0;
	  this.gatedfrom = [];

	  if (typeof this.type == 'undefined')
	  {
	    if (fromLayer == toLayer)
	      this.type = Layer.connectionType.ONE_TO_ONE;
	    else
	      this.type = Layer.connectionType.ALL_TO_ALL;
	  }

	  if (this.type == Layer.connectionType.ALL_TO_ALL ||
	      this.type == Layer.connectionType.ALL_TO_ELSE) {
	    for (var here in this.from.list) {
	      for (var there in this.to.list) {
	        var from = this.from.list[here];
	        var to = this.to.list[there];
	        if(this.type == Layer.connectionType.ALL_TO_ELSE && from == to)
	          continue;
	        var connection = from.project(to, weights);

	        this.connections[connection.ID] = connection;
	        this.size = this.list.push(connection);
	      }
	    }
	  } else if (this.type == Layer.connectionType.ONE_TO_ONE) {

	    for (var neuron in this.from.list) {
	      var from = this.from.list[neuron];
	      var to = this.to.list[neuron];
	      var connection = from.project(to, weights);

	      this.connections[connection.ID] = connection;
	      this.size = this.list.push(connection);
	    }
	  }

	  fromLayer.connectedTo.push(this);
	}

	// types of connections
	Layer.connectionType = {};
	Layer.connectionType.ALL_TO_ALL = "ALL TO ALL";
	Layer.connectionType.ONE_TO_ONE = "ONE TO ONE";
	Layer.connectionType.ALL_TO_ELSE = "ALL TO ELSE";

	(function() {
	  var connections = 0;
	  Layer.connection.uid = function() {
	    return connections++;
	  }
	})();

	/*******************************************************************************************
	                                         NETWORK
	*******************************************************************************************/

	function Network(layers) {
	  if (typeof layers != 'undefined') {
	    this.layers = {
	      input:  layers.input || null,
	      hidden: layers.hidden || [],
	      output: layers.output || null
	    };
	    this.optimized = null;
	  }
	}
	Network.prototype = {

	  // feed-forward activation of all the layers to produce an ouput
	  activate: function(input) {
	    if (this.optimized === false)
	    {
	      this.layers.input.activate(input);
	      for (var i = 0; i < this.layers.hidden.length; i++)
	        this.layers.hidden[i].activate();
	      return this.layers.output.activate();
	    }
	    else
	    {
	      if (this.optimized == null)
	        this.optimize();
	      return this.optimized.activate(input);
	    }
	  },

	  // back-propagate the error thru the network
	  propagate: function(rate, target) {
	    if (this.optimized === false)
	    {
	      this.layers.output.propagate(rate, target);
	      for (var i = this.layers.hidden.length - 1; i >= 0; i--)
	        this.layers.hidden[i].propagate(rate);
	    }
	    else
	    {
	      if (this.optimized == null)
	        this.optimize();
	      this.optimized.propagate(rate, target);
	    }
	  },

	  // project a connection to another unit (either a network or a layer)
	  project: function(unit, type, weights) {
	    if (this.optimized)
	      this.optimized.reset();

	    if (unit instanceof Network)
	      return this.layers.output.project(unit.layers.input, type, weights);

	    if (unit instanceof Layer)
	      return this.layers.output.project(unit, type, weights);

	    throw new Error("Invalid argument, you can only project connections to LAYERS and NETWORKS!");
	  },

	  // let this network gate a connection
	  gate: function(connection, type) {
	    if (this.optimized)
	      this.optimized.reset();
	    this.layers.output.gate(connection, type);
	  },

	  // clear all elegibility traces and extended elegibility traces (the network forgets its context, but not what was trained)
	  clear: function() {
	    this.restore();

	    var inputLayer = this.layers.input,
	      outputLayer = this.layers.output;

	    inputLayer.clear();
	    for (var i = 0; i < this.layers.hidden.length; i++) {
	      this.layers.hidden[i].clear();
	    }
	    outputLayer.clear();

	    if (this.optimized)
	      this.optimized.reset();
	  },

	  // reset all weights and clear all traces (ends up like a new network)
	  reset: function() {
	    this.restore();

	    var inputLayer = this.layers.input,
	      outputLayer = this.layers.output;

	    inputLayer.reset();
	    for (var i = 0; i < this.layers.hidden.length; i++) {
	      this.layers.hidden[i].reset();
	    }
	    outputLayer.reset();

	    if (this.optimized)
	      this.optimized.reset();
	  },

	  // hardcodes the behaviour of the whole network into a single optimized function
	  optimize: function() {
	    var that = this;
	    var optimized = {};
	    var neurons = this.neurons();

	    for (var i = 0; i < neurons.length; i++) {
	      var neuron = neurons[i].neuron;
	      var layer = neurons[i].layer;
	      while (neuron.neuron)
	        neuron = neuron.neuron;
	      optimized = neuron.optimize(optimized, layer);
	    }

	    for (var i = 0; i < optimized.propagation_sentences.length; i++)
	      optimized.propagation_sentences[i].reverse();
	    optimized.propagation_sentences.reverse();

	    var hardcode = "";
	    hardcode += "var F = Float64Array ? new Float64Array(" + optimized.memory +
	      ") : []; ";
	    for (var i in optimized.variables)
	      hardcode += "F[" + optimized.variables[i].id + "] = " + (optimized.variables[
	        i].value || 0) + "; ";
	    hardcode += "var activate = function(input){\n";
	    for (var i = 0; i < optimized.inputs.length; i++)
	      hardcode += "F[" + optimized.inputs[i] + "] = input[" + i + "]; ";
	    for (var i = 0; i < optimized.activation_sentences.length; i++) {
	      if (optimized.activation_sentences[i].length > 0) {
	        for (var j = 0; j < optimized.activation_sentences[i].length; j++) {
	          hardcode += optimized.activation_sentences[i][j].join(" ");
	          hardcode += optimized.trace_sentences[i][j].join(" ");
	        }
	      }
	    }
	    hardcode += " var output = []; "
	    for (var i = 0; i < optimized.outputs.length; i++)
	      hardcode += "output[" + i + "] = F[" + optimized.outputs[i] + "]; ";
	    hardcode += "return output; }; "
	    hardcode += "var propagate = function(rate, target){\n";
	    hardcode += "F[" + optimized.variables.rate.id + "] = rate; ";
	    for (var i = 0; i < optimized.targets.length; i++)
	      hardcode += "F[" + optimized.targets[i] + "] = target[" + i + "]; ";
	    for (var i = 0; i < optimized.propagation_sentences.length; i++)
	      for (var j = 0; j < optimized.propagation_sentences[i].length; j++)
	        hardcode += optimized.propagation_sentences[i][j].join(" ") + " ";
	    hardcode += " };\n";
	    hardcode +=
	      "var ownership = function(memoryBuffer){\nF = memoryBuffer;\nthis.memory = F;\n};\n";
	    hardcode +=
	      "return {\nmemory: F,\nactivate: activate,\npropagate: propagate,\nownership: ownership\n};";
	    hardcode = hardcode.split(";").join(";\n");

	    var constructor = new Function(hardcode);

	    var network = constructor();
	    network.data = {
	      variables: optimized.variables,
	      activate: optimized.activation_sentences,
	      propagate: optimized.propagation_sentences,
	      trace: optimized.trace_sentences,
	      inputs: optimized.inputs,
	      outputs: optimized.outputs,
	      check_activation: this.activate,
	      check_propagation: this.propagate
	    }

	    network.reset = function() {
	      if (that.optimized) {
	        that.optimized = null;
	        that.activate = network.data.check_activation;
	        that.propagate = network.data.check_propagation;
	      }
	    }

	    this.optimized = network;
	    this.activate = network.activate;
	    this.propagate = network.propagate;
	  },

	  // restores all the values from the optimized network the their respective objects in order to manipulate the network
	  restore: function() {
	    if (!this.optimized)
	      return;

	    var optimized = this.optimized;

	    var getValue = function() {
	      var args = Array.prototype.slice.call(arguments);

	      var unit = args.shift();
	      var prop = args.pop();

	      var id = prop + '_';
	      for (var property in args)
	        id += args[property] + '_';
	      id += unit.ID;

	      var memory = optimized.memory;
	      var variables = optimized.data.variables;

	      if (id in variables)
	        return memory[variables[id].id];
	      return 0;
	    }

	    var list = this.neurons();

	    // link id's to positions in the array
	    for (var i = 0; i < list.length; i++) {
	      var neuron = list[i].neuron;
	      while (neuron.neuron)
	        neuron = neuron.neuron;

	      neuron.state = getValue(neuron, 'state');
	      neuron.old = getValue(neuron, 'old');
	      neuron.activation = getValue(neuron, 'activation');
	      neuron.bias = getValue(neuron, 'bias');

	      for (var input in neuron.trace.elegibility)
	        neuron.trace.elegibility[input] = getValue(neuron, 'trace',
	          'elegibility', input);

	      for (var gated in neuron.trace.extended)
	        for (var input in neuron.trace.extended[gated])
	          neuron.trace.extended[gated][input] = getValue(neuron, 'trace',
	            'extended', gated, input);

	      // get connections
	      for (var j in neuron.connections.projected) {
	        var connection = neuron.connections.projected[j];
	        connection.weight = getValue(connection, 'weight');
	        connection.gain = getValue(connection, 'gain');
	      }
	    }
	  },

	  // returns all the neurons in the network
	  neurons: function() {
	    var neurons = [];

	    var inputLayer = this.layers.input.neurons(),
	      outputLayer = this.layers.output.neurons();

	    for (var i = 0; i < inputLayer.length; i++) {
	      neurons.push({
	        neuron: inputLayer[i],
	        layer: 'input'
	      });
	    }

	    for (var i = 0; i < this.layers.hidden.length; i++) {
	      var hiddenLayer = this.layers.hidden[i].neurons();
	      for (var j = 0; j < hiddenLayer.length; j++)
	        neurons.push({
	          neuron: hiddenLayer[j],
	          layer: i
	        });
	    }

	    for (var i = 0; i < outputLayer.length; i++) {
	      neurons.push({
	        neuron: outputLayer[i],
	        layer: 'output'
	      });
	    }

	    return neurons;
	  },

	  // returns number of inputs of the network
	  inputs: function() {
	    return this.layers.input.size;
	  },

	  // returns number of outputs of hte network
	  outputs: function() {
	    return this.layers.output.size;
	  },

	  // sets the layers of the network
	  set: function(layers) {
	    this.layers = {
	      input:  layers.input || null,
	      hidden: layers.hidden || [],
	      output: layers.output || null
	    };
	    if (this.optimized)
	      this.optimized.reset();
	  },

	  setOptimize: function(bool){
	    this.restore();
	    if (this.optimized)
	      this.optimized.reset();
	    this.optimized = bool? null : false;
	  },

	  // returns a json that represents all the neurons and connections of the network
	  toJSON: function(ignoreTraces) {
	    this.restore();

	    var list = this.neurons();
	    var neurons = [];
	    var connections = [];

	    // link id's to positions in the array
	    var ids = {};
	    for (var i = 0; i < list.length; i++) {
	      var neuron = list[i].neuron;
	      while (neuron.neuron)
	        neuron = neuron.neuron;
	      ids[neuron.ID] = i;

	      var copy = {
	        trace: {
	          elegibility: {},
	          extended: {}
	        },
	        state: neuron.state,
	        old: neuron.old,
	        activation: neuron.activation,
	        bias: neuron.bias,
	        layer: list[i].layer
	      };

	      copy.transfer = neuron.transfer == Neuron.transfer.LOGISTIC ? "LOGISTIC" :
	        neuron.transfer == Neuron.transfer.TANH ? "TANH" :
	        neuron.transfer == Neuron.transfer.IDENTITY ? "IDENTITY" :
	        neuron.transfer == Neuron.transfer.HLIM ? "HLIM" :
	        neuron.transfer == Neuron.transfer.RELU ? "RELU" :
	        null;

	      neurons.push(copy);
	    }

	    for(var i = 0; i < list.length; i++){
	      var neuron = list[i].neuron;
	      while (neuron.neuron)
	        neuron = neuron.neuron;

	      for (var j in neuron.connections.projected) {
	        var connection = neuron.connections.projected[j];
	        connections.push({
	          from: ids[connection.from.ID],
	          to: ids[connection.to.ID],
	          weight: connection.weight,
	          gater: connection.gater ? ids[connection.gater.ID] : null,
	        });
	      }
	      if (neuron.selfconnected()) {
	        connections.push({
	          from: ids[neuron.ID],
	          to: ids[neuron.ID],
	          weight: neuron.selfconnection.weight,
	          gater: neuron.selfconnection.gater ? ids[neuron.selfconnection.gater.ID] : null,
	        });
	      }
	    }

	    return {
	      neurons: neurons,
	      connections: connections
	    }
	  },

	  // export the topology into dot language which can be visualized as graphs using dot
	  /* example: ... console.log(net.toDotLang());
	              $ node example.js > example.dot
	              $ dot example.dot -Tpng > out.png
	  */
	  toDot: function(edgeConnection) {
	    if (! typeof edgeConnection)
	      edgeConnection = false;
	    var code = "digraph nn {\n    rankdir = BT\n";
	    var layers = [this.layers.input].concat(this.layers.hidden, this.layers.output);
	    for (var i = 0; i < layers.length; i++) {
	      for (var j = 0; j < layers[i].connectedTo.length; j++) { // projections
	        var connection = layers[i].connectedTo[j];
	        var layerTo = connection.to;
	        var size = connection.size;
	        var layerID = layers.indexOf(layers[i]);
	        var layerToID = layers.indexOf(layerTo);
	        /* http://stackoverflow.com/questions/26845540/connect-edges-with-graph-dot
	         * DOT does not support edge-to-edge connections
	         * This workaround produces somewhat weird graphs ...
	        */
	        if ( edgeConnection) {
	          if (connection.gatedfrom.length) {
	            var fakeNode = "fake" + layerID + "_" + layerToID;
	            code += "    " + fakeNode +
	              " [label = \"\", shape = point, width = 0.01, height = 0.01]\n";
	            code += "    " + layerID + " -> " + fakeNode + " [label = " + size + ", arrowhead = none]\n";
	            code += "    " + fakeNode + " -> " + layerToID + "\n";
	          } else
	            code += "    " + layerID + " -> " + layerToID + " [label = " + size + "]\n";
	          for (var from in connection.gatedfrom) { // gatings
	            var layerfrom = connection.gatedfrom[from].layer;
	            var layerfromID = layers.indexOf(layerfrom);
	            code += "    " + layerfromID + " -> " + fakeNode + " [color = blue]\n";
	          }
	        } else {
	          code += "    " + layerID + " -> " + layerToID + " [label = " + size + "]\n";
	          for (var from in connection.gatedfrom) { // gatings
	            var layerfrom = connection.gatedfrom[from].layer;
	            var layerfromID = layers.indexOf(layerfrom);
	            code += "    " + layerfromID + " -> " + layerToID + " [color = blue]\n";
	          }
	        }
	      }
	    }
	    code += "}\n";
	    return {
	      code: code,
	      link: "https://chart.googleapis.com/chart?chl=" + escape(code.replace("/ /g", "+")) + "&cht=gv"
	    }
	  },

	  // returns a function that works as the activation of the network and can be used without depending on the library
	  standalone: function() {
	    if (!this.optimized)
	      this.optimize();

	    var data = this.optimized.data;

	    // build activation function
	    var activation = "function (input) {\n";

	    // build inputs
	    for (var i = 0; i < data.inputs; i++)
	      activation += "F[" + data.inputs[i] + "] = input[" + i + "];\n";

	    // build network activation
	    for (var i = 0; i < data.activate.length; i++) { // shouldn't this be layer?
	      for (var j = 0; j <  data.activate[i].length; j++)
	        activation += data.activate[i][j].join('') + "\n";
	    }

	    // build outputs
	    activation += "var output = [];\n";
	    for (var i = 0; i < data.outputs.length; i++)
	      activation += "output[" + i + "] = F[" + data.outputs[i] + "];\n";
	    activation += "return output;\n}";

	    // reference all the positions in memory
	    var memory = activation.match(/F\[(\d+)\]/g);
	    var dimension = 0;
	    var ids = {};

	    for (var i = 0; i < memory.length; i++) {
	      var tmp = memory[i].match(/\d+/)[0];
	      if (!(tmp in ids)) {
	        ids[tmp] = dimension++;
	      }
	    }
	    var hardcode = "F = {\n";

	    for (var i in ids)
	      hardcode += ids[i] + ": " + this.optimized.memory[i] + ",\n";
	    hardcode = hardcode.substring(0, hardcode.length - 2) + "\n};\n";
	    hardcode = "var run = " + activation.replace(/F\[(\d+)]/g, function(
	      index) {
	      return 'F[' + ids[index.match(/\d+/)[0]] + ']'
	    }).replace("{\n", "{\n" + hardcode + "") + ";\n";
	    hardcode += "return run";

	    // return standalone function
	    return new Function(hardcode)();
	  },


	  // Return a HTML5 WebWorker specialized on training the network stored in `memory`.
	  // Train based on the given dataSet and options.
	  // The worker returns the updated `memory` when done.
	  worker: function(memory, set, options) {

	    // Copy the options and set defaults (options might be different for each worker)
	    var workerOptions = {};
	    if(options) workerOptions = options;
	    workerOptions.rate = options.rate || .2;
	    workerOptions.iterations = options.iterations || 100000;
	    workerOptions.error = options.error || .005;
	    workerOptions.cost = options.cost || null;
	    workerOptions.crossValidate = options.crossValidate || null;

	    // Cost function might be different for each worker
	    costFunction = "var cost = " + (options && options.cost || this.cost || Trainer.cost.MSE) + ";\n";
	    var workerFunction = Network.getWorkerSharedFunctions();
	    workerFunction = workerFunction.replace(/var cost = options && options\.cost \|\| this\.cost \|\| Trainer\.cost\.MSE;/g, costFunction);

	    // Set what we do when training is finished
	    workerFunction = workerFunction.replace('return results;',
	                      'postMessage({action: "done", message: results, memoryBuffer: F}, [F.buffer]);');

	    // Replace log with postmessage
	    workerFunction = workerFunction.replace("console.log('iterations', iterations, 'error', error, 'rate', currentRate)",
	              "postMessage({action: 'log', message: {\n" +
	                  "iterations: iterations,\n" +
	                  "error: error,\n" +
	                  "rate: currentRate\n" +
	                "}\n" +
	              "})");

	    // Replace schedule with postmessage
	    workerFunction = workerFunction.replace("abort = this.schedule.do({ error: error, iterations: iterations, rate: currentRate })",
	              "postMessage({action: 'schedule', message: {\n" +
	                  "iterations: iterations,\n" +
	                  "error: error,\n" +
	                  "rate: currentRate\n" +
	                "}\n" +
	              "})");

	    if (!this.optimized)
	      this.optimize();

	    var hardcode = "var inputs = " + this.optimized.data.inputs.length + ";\n";
	    hardcode += "var outputs = " + this.optimized.data.outputs.length + ";\n";
	    hardcode += "var F =  new Float64Array([" + this.optimized.memory.toString() + "]);\n";
	    hardcode += "var activate = " + this.optimized.activate.toString() + ";\n";
	    hardcode += "var propagate = " + this.optimized.propagate.toString() + ";\n";
	    hardcode +=
	        "onmessage = function(e) {\n" +
	          "if (e.data.action == 'startTraining') {\n" +
	            "train(" + JSON.stringify(set) + "," + JSON.stringify(workerOptions) + ");\n" +
	          "}\n" +
	        "}";

	    var workerSourceCode = workerFunction + '\n' + hardcode;
	    var blob = new Blob([workerSourceCode]);
	    var blobURL = window.URL.createObjectURL(blob);

	    return new Worker(blobURL);
	  },

	  // returns a copy of the network
	  clone: function() {
	    return Network.fromJSON(this.toJSON());
	  }
	};

	/**
	 * Creates a static String to store the source code of the functions
	 *  that are identical for all the workers (train, _trainSet, test)
	 *
	 * @return {String} Source code that can train a network inside a worker.
	 * @static
	 */
	Network.getWorkerSharedFunctions = function() {
	  // If we already computed the source code for the shared functions
	  if(typeof Network._SHARED_WORKER_FUNCTIONS !== 'undefined')
	    return Network._SHARED_WORKER_FUNCTIONS;

	  // Otherwise compute and return the source code
	  // We compute them by simply copying the source code of the train, _trainSet and test functions
	  //  using the .toString() method

	  // Load and name the train function
	  var train_f = Trainer.prototype.train.toString();
	  train_f = train_f.replace('function (set', 'function train(set') + '\n';

	  // Load and name the _trainSet function
	  var _trainSet_f = Trainer.prototype._trainSet.toString().replace(/this.network./g, '');
	  _trainSet_f = _trainSet_f.replace('function (set', 'function _trainSet(set') + '\n';
	  _trainSet_f = _trainSet_f.replace('this.crossValidate', 'crossValidate');
	  _trainSet_f = _trainSet_f.replace('crossValidate = true', 'crossValidate = { }');

	  // Load and name the test function
	  var test_f = Trainer.prototype.test.toString().replace(/this.network./g, '');
	  test_f = test_f.replace('function (set', 'function test(set') + '\n';

	  return Network._SHARED_WORKER_FUNCTIONS = train_f + _trainSet_f + test_f;
	};

	// rebuild a network that has been stored in a json using the method toJSON()
	Network.fromJSON = function(json) {
	  var neurons = [];

	  var layers = {
	    input: new Layer(),
	    hidden: [],
	    output: new Layer()
	  };

	  for (var i = 0; i < json.neurons.length; i++) {
	    var config = json.neurons[i];

	    var neuron = new Neuron();
	    neuron.trace.elegibility = {};
	    neuron.trace.extended = {};
	    neuron.state = config.state;
	    neuron.old = config.old;
	    neuron.activation = config.activation;
	    neuron.bias = config.bias;
	    neuron.transfer = config.transfer in Neuron.transfer ? Neuron.transfer[config.transfer] : Neuron.transfer.LOGISTIC;
	    neurons.push(neuron);

	    if (config.layer == 'input')
	      layers.input.add(neuron);
	    else if (config.layer == 'output')
	      layers.output.add(neuron);
	    else {
	      if (typeof layers.hidden[config.layer] == 'undefined')
	        layers.hidden[config.layer] = new Layer();
	      layers.hidden[config.layer].add(neuron);
	    }
	  }

	  for (var i = 0; i < json.connections.length; i++) {
	    var config = json.connections[i];
	    var from = neurons[config.from];
	    var to = neurons[config.to];
	    var weight = config.weight;
	    var gater = neurons[config.gater];

	    var connection = from.project(to, weight);
	    if (gater)
	      gater.gate(connection);
	  }

	  return new Network(layers);
	};


	/*******************************************************************************************
	                                        TRAINER
	*******************************************************************************************/

	//+ Jonas Raoni Soares Silva
	//@ http://jsfromhell.com/array/shuffle [v1.0]
	function shuffleInplace(o) { //v1.0
	  for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
	  return o;
	};

	function Trainer(network, options) {
	  options = options || {};
	  this.network = network;
	  this.rate = options.rate || .2;
	  this.iterations = options.iterations || 100000;
	  this.error = options.error || .005;
	  this.cost = options.cost || null;
	  this.crossValidate = options.crossValidate || null;
	}

	Trainer.prototype = {

	  // trains any given set to a network
	  train: function(set, options) {
	    var error = 1;
	    var iterations = bucketSize = 0;
	    var abort = false;
	    var currentRate;
	    var cost = options && options.cost || this.cost || Trainer.cost.MSE;
	    var crossValidate = false, testSet, trainSet;

	    var start = Date.now();

	    if (options) {
	      if (options.iterations)
	        this.iterations = options.iterations;
	      if (options.error)
	        this.error = options.error;
	      if (options.rate)
	        this.rate = options.rate;
	      if (options.cost)
	        this.cost = options.cost;
	      if (options.schedule)
	        this.schedule = options.schedule;
	      if (options.customLog){
	        // for backward compatibility with code that used customLog
	        console.log('Deprecated: use schedule instead of customLog')
	        this.schedule = options.customLog;
	      }
	      if (this.crossValidate || options.crossValidate) {
	        if(!this.crossValidate) this.crossValidate = {};
	        crossValidate = true;
	        if (options.crossValidate.testSize)
	          this.crossValidate.testSize = options.crossValidate.testSize;
	        if (options.crossValidate.testError)
	          this.crossValidate.testError = options.crossValidate.testError;
	      }
	    }

	    currentRate = this.rate;
	    if(Array.isArray(this.rate)) {
	      var bucketSize = Math.floor(this.iterations / this.rate.length);
	    }

	    if(crossValidate) {
	      var numTrain = Math.ceil((1 - this.crossValidate.testSize) * set.length);
	      trainSet = set.slice(0, numTrain);
	      testSet = set.slice(numTrain);
	    }

	    var lastError = 0;
	    while ((!abort && iterations < this.iterations && error > this.error)) {
	      if (crossValidate && error <= this.crossValidate.testError) {
	        break;
	      }

	      var currentSetSize = set.length;
	      error = 0;
	      iterations++;

	      if(bucketSize > 0) {
	        var currentBucket = Math.floor(iterations / bucketSize);
	        currentRate = this.rate[currentBucket] || currentRate;
	      }

	      if(typeof this.rate === 'function') {
	        currentRate = this.rate(iterations, lastError);
	      }

	      if (crossValidate) {
	        this._trainSet(trainSet, currentRate, cost);
	        error += this.test(testSet).error;
	        currentSetSize = 1;
	      } else {
	        error += this._trainSet(set, currentRate, cost);
	        currentSetSize = set.length;
	      }

	      // check error
	      error /= currentSetSize;
	      lastError = error;

	      if (options) {
	        if (this.schedule && this.schedule.every && iterations %
	          this.schedule.every == 0)
	          abort = this.schedule.do({ error: error, iterations: iterations, rate: currentRate });
	        else if (options.log && iterations % options.log == 0) {
	          console.log('iterations', iterations, 'error', error, 'rate', currentRate);
	        };
	        if (options.shuffle)
	          shuffleInplace(set);
	      }
	    }

	    var results = {
	      error: error,
	      iterations: iterations,
	      time: Date.now() - start
	    };

	    return results;
	  },

	  // preforms one training epoch and returns the error (private function used in this.train)
	  _trainSet: function(set, currentRate, costFunction) {
	    var errorSum = 0;
	    for (var i = 0; i < set.length; i++) {
	      var input = set[i].input;
	      var target = set[i].output;

	      var output = this.network.activate(input);
	      this.network.propagate(currentRate, target);

	      errorSum += costFunction(target, output);
	    }
	    return errorSum;
	  },

	  // tests a set and returns the error and elapsed time
	  test: function(set, options) {
	    var error = 0;
	    var input, output, target;
	    var cost = options && options.cost || this.cost || Trainer.cost.MSE;

	    var start = Date.now();

	    for (var i = 0; i < set.length; i++) {
	      input = set[i].input;
	      target = set[i].output;
	      output = this.network.activate(input);
	      error += cost(target, output);
	    }

	    error /= set.length;

	    var results = {
	      error: error,
	      time: Date.now() - start
	    };

	    return results;
	  },
	};

	// Built-in cost functions
	Trainer.cost = {
	  // Eq. 9
	  MSE: function(target, output)
	  {
	    var mse = 0;
	    for (var i = 0; i < output.length; i++)
	      mse += Math.pow(target[i] - output[i], 2);
	    return mse / output.length;
	  },
	}



	/*******************************************************************************************
	                                        ARCHITECT
	*******************************************************************************************/

	// Collection of useful built-in architectures
	var Architect = {

	  // Multilayer Perceptron
	  Perceptron: function Perceptron() {
	    var args = Array.prototype.slice.call(arguments); // convert arguments to Array
	    if (args.length < 3)
	      throw new Error("not enough layers (minimum 3) !!");

	    var inputs = args.shift(); // first argument
	    var outputs = args.pop(); // last argument
	    var layers = args; // all the arguments in the middle

	    var input = new Layer(inputs);
	    var hidden = [];
	    var output = new Layer(outputs);

	    var previous = input;

	    // generate hidden layers
	    for (var i = 0; i < layers.length; i++) {
	      var size = layers[i];
	      var layer = new Layer(size);
	      hidden.push(layer);
	      previous.project(layer);
	      previous = layer;
	    }
	    previous.project(output);

	    // set layers of the neural network
	    this.set({
	      input: input,
	      hidden: hidden,
	      output: output
	    });
	  },
	}

	// Extend prototype chain (so every architectures is an instance of Network)
	for (var architecture in Architect) {
	  Architect[architecture].prototype = new Network();
	  Architect[architecture].prototype.constructor = Architect[architecture];
	}
