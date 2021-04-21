let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var object1Samples=0, object2Samples=0, object3Samples=0, object4Samples=0, object5Samples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);
    
  // In the space below create a neural network that can classify 5 objects. The first layer
  // of your network should be a flatten layer that takes as input the output
  // from the pre-trained MobileNet model. Since we have 5 classes, your output
  // layer should have 5 units and a softmax activation function. You are free
  // to use as many hidden layers and neurons as you like.  
  // It is suggested to use ReLu activation functions where applicable.
    
  model = tf.sequential({
    layers: [
        tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
        tf.layers.dense({ units: 160, activation: 'relu'}),
        tf.layers.dense({ units: 5, activation: 'softmax'})
        ]
  });
      
  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  const optimizer = tf.train.adam(0.0001);
            
  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
 
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			object1Samples++;
			document.getElementById("object1samples").innerText = "Object1 samples:" + object1Samples;
			break;
		case "1":
			object2Samples++;
			document.getElementById("object2samples").innerText = "Object2 samples:" + object2Samples;
			break;
		case "2":
			object3Samples++;
			document.getElementById("object3samples").innerText = "Object3 samples:" + object3Samples;
			break;  
		case "3":
			object4Samples++;
			document.getElementById("object4samples").innerText = "Object4 samples:" + object4Samples;
			break;
        // Add a case for lizard samples.
        // HINT: Look at the previous cases.
        case "4":
			object5Samples++;
			document.getElementById("object5samples").innerText = "Object5 samples:" + object5Samples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Object1";
			break;
		case 1:
			predictionText = "I see Object2";
			break;
		case 2:
			predictionText = "I see Object3";
			break;
		case 3:
			predictionText = "I see Object4";
			break;
        // Add a case for lizard samples.
        // HINT: Look at the previous cases.
        case 4:
			predictionText = "I see Object5";
			break;
	
            
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
	alert("Training Done!")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


function saveModel(){
    model.save('downloads://my_model');
}


async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();