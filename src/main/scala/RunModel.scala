import breeze.linalg._
import breeze.stats._

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

//    val len = labels.length
    val len = 3000
    val cut = (0.6 * len).toInt
    val (x, y, xCV, yCV) = (features(0 to cut-1, ::), labels.take(cut), features(cut to len-1, ::), labels.slice(cut, len))

    val classifierKNN = new KNN
    classifierKNN.k = 3

    val startTrainTime = System.nanoTime()
    classifierKNN.train(x, y)
    val trainTime = System.nanoTime()
    println("Training set length = "+classifierKNN.trainingLabels.length)
    println("Test set length = "+yCV.length)

    def printHist() = {
      val histTrain = y.groupBy(x => x).mapValues(_.size)
      println("Training set histogram: "+histTrain + "   Mean="+mean(histTrain.values.map(_.toDouble))+ "   STD="+stddev(histTrain.values.map(_.toDouble)))
      val histTest = yCV.groupBy(x => x).mapValues(_.size)
      println("Test set histogram: " + histTest + "   Mean=" + mean(histTest.values.map(_.toDouble)) + "   STD=" + stddev(histTest.values.map(_.toDouble)))
    }

    val startPredictionTime = System.nanoTime()
    val result = classifierKNN.predict(xCV)
//    val resultDist = classifierKNN.predictDist(xCV)
    val predictionTime = System.nanoTime()

    // look at error vs k
//    List.range(1, 10).map( x => {
//      classifierKNN.k = x
//      val result = classifierKNN.getKNNFromDist(resultDist)
//      // println(result.zip(yCV))
//      println("k="+x.toString+"   error="+error(result, yCV).toString)
//    })

    // print run time
    val lTime = (loadTime - startLoadTime)*1.0/1000000000
    val tTime = (trainTime - startTrainTime)*1.0/1000000000
    val pTime = (predictionTime - startTrainTime)*1.0/1000000000
    println("Load time = "+lTime+"s   Trainin time = "+tTime+"s   Prediction time = "+pTime+"s")
    println("k="+classifierKNN.k+"   error="+error(result, yCV).toString)
  }

  def error(predictions: List[Int], truth: List[Int]): Double = {
    val wrong = truth.zip(predictions).filter( x => x._1 != x._2)
    wrong.size.toDouble / truth.size
  }
}
