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
    println(features.cols)
  }
}
