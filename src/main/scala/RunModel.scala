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
    val startLoadTime = System.nanoTime()
    val (features,labels) = loader.loadMNISTData(path)
    val loadTime = System.nanoTime()
//    val path = "data/iris.data"
//    val (features,labels) = loader.loadIrisData(path)

//    val len = labels.length
    val len = 5000
    val cut = (0.6 * len).toInt
    val (x, y, xCV, yCV) = (features(0 to cut-1, ::), labels.take(cut), features(cut to len-1, ::), labels.slice(cut, len))

    val classifierKNN = new KNN
    val startTrainTime = System.nanoTime()
    classifierKNN.train(x, y)
    val trainTime = System.nanoTime()
    println("Training set length = "+classifierKNN.trainingLabels.length)
    println("Test set length = "+yCV.length)

    val startPredictionTime = System.nanoTime()
//    val result = classifierKNN.predict(xCV)
    val resultDist = classifierKNN.predictDist(xCV)
    val predictionTime = System.nanoTime()

    List.range(1, 10).map( x => {
      classifierKNN.k = x
      val result = classifierKNN.getKNNFromDist(resultDist)
      // println(result.zip(yCV))
      println("k="+x.toString+"   error="+error(result, yCV).toString)
    })

    // print run time
    val lTime = (loadTime - startLoadTime)*1.0/1000000000
    val tTime = (trainTime - startTrainTime)*1.0/1000000000
    val pTime = (predictionTime - startTrainTime)*1.0/1000000000
    println("Load time = "+lTime+"s   Trainin time = "+tTime+"s   Prediction time = "+pTime+"s")
  }

  def error(predictions: List[Int], truth: List[Int]): Double = {
    val wrong = truth.zip(predictions).filter( x => x._1 != x._2)
    wrong.size.toDouble / truth.size
  }
}
