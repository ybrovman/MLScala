import breeze.linalg._

object RunModel {
  def main(args: Array[String]) {
    println("Running model...")
    val myLoader = new Loader

//    runLogisticRegression(myLoader)
    runKNN(myLoader)
  }

  def runLogisticRegression(loader: Loader) = {
    val path = "data/iris.data"
    val (features,labels) = loader.loadIrisData(path)
    println(features.rows)
//    labels.foreach(println)
  }

  def runKNN(loader: Loader) = {
    val path = "data/MNIST.csv"
    val (features,labels) = loader.loadMNISTData(path)

    val len = labels.length
    val cut = (0.6 * len).toInt
    val (x, y, xCV, yCV) = (features(0 to cut-1, ::), labels(0 to cut-1), features(cut to len-1, ::), labels(cut to len-1))

    val classifierKNN = new KNN
    classifierKNN.train(x, y)
    println("Training set length = "+classifierKNN.trainingLabels.length)
    println("Test set length = "+yCV.length)

    classifierKNN.predict(xCV)
  }
}
