const tfjs = require("@tensorflow/tfjs-node");

// fungsi untuk load model tensorflow.js
function loadModel() {
	const modelUrl = "file://models/model.json";
	// Method loadLayersModel() bertujuan memuat (load) suatu model neural network.
	// loadLayersModel() menerima argumen berupa url dengan skema “localstorage://” dan “indexeddb://” jika model disimpan dalam browser.
	// Selain itu, skema “http://” dan “https://” dapat digunakan jika model disimpan dalam suatu storage, seperti Object Storage atau Google Cloud Storage buckets.
	// loadLayersModel() mengembalikan Promise<tf.LayersModel>.
	// Jadi, ketika Anda memanggil loadLayersModel() harus digunakan dalam fungsi asynchronous.
	return tfjs.loadLayersModel(modelUrl);
}

// Fungsi kedua adalah predict() untuk memprediksi data (berupa imageBuffer) dengan model yang telah di-load.
function predict(model, imageBuffer) {
	const tensor = tfjs.node
        .decodeJpeg(imageBuffer)
        .resizeNearestNeighbor([150, 150])
        .expandDims()
        .toFloat();

	return model.predict(tensor).data();
}

module.exports = { loadModel, predict };
