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
//    val path = "data/iris.data"
//    val (features,labels) = loader.loadIrisData(path)

//    val len = labels.length
    val len = 1000
    val cut = (0.6 * len).toInt
    val (x, y, xCV, yCV) = (features(0 to cut-1, ::), labels.take(cut), features(cut to len-1, ::), labels.slice(cut, len))

    val classifierKNN = new KNN
    classifierKNN.train(x, y)
    println("Training set length = "+classifierKNN.trainingLabels.length)
    println("Test set length = "+yCV.length)

    List.range(1, 10).map( x => {
      classifierKNN.k = x
      val result = classifierKNN.predict(xCV)
      // println(result.zip(yCV))
      println("k="+x.toString+"   error="+error(result, yCV).toString)
    })
  }

  def error(predictions: List[Int], truth: List[Int]): Double = {
    val wrong = truth.zip(predictions).filter( x => x._1 != x._2)
    wrong.size.toDouble / truth.size
  }
}
