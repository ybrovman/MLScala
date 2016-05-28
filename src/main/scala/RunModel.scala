import breeze.linalg._
import breeze.stats._

object RunModel {
  def main(args: Array[String]) {
    println("Running model...")
    val myLoader = new Loader

    runLogisticRegression(myLoader)
//    runKNN(myLoader)
  }

  def runLogisticRegression(loader: Loader) = {
//    val path = "data/iris.data"
//    val (features,labels) = loader.loadIrisData(path)
    val path = "data/MNIST.csv"
    val (features,labels) = loader.loadMNISTDataBinary(path, 1, 8)

    val len = labels.length
    val cut = (0.6 * len).toInt
    val (x, y, xCV, yCV) = (features(0 to cut-1, ::), labels.take(cut), features(cut to len-1, ::), labels.slice(cut, len))

    val classifierLogit = new LogistricRegression

    // for MNIST data
    classifierLogit.alpha = 0.000005
    classifierLogit.convergenceTolerance = 0.001

    classifierLogit.train(x, y)
    val result = classifierLogit.predict(xCV)
    println(result.zip(yCV))
    println("error="+error(result, yCV).toString)
    println("Final beta = "+classifierLogit.beta)

    def printHist() = {
      val histTrain = y.groupBy(x => x).mapValues(_.size)
      println("Training set histogram: "+histTrain + "   Mean="+mean(histTrain.values.map(_.toDouble))+ "   STD="+stddev(histTrain.values.map(_.toDouble)))
      val histTest = yCV.groupBy(x => x).mapValues(_.size)
      println("Test set histogram: " + histTest + "   Mean=" + mean(histTest.values.map(_.toDouble)) + "   STD=" + stddev(histTest.values.map(_.toDouble)))
      val histPred = result.groupBy(x => x).mapValues(_.size)
      println("Prediction set histogram: " + histPred + "   Mean=" + mean(histPred.values.map(_.toDouble)) + "   STD=" + stddev(histPred.values.map(_.toDouble)))
    }
    printHist
//    val lambdas = List(100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0)
//    val errors = lambdas.map( lambda => {
//      classifierLogit.lambda = lambda
//      classifierLogit.train(x, y)
//      val result = classifierLogit.predict(xCV)
//      println("error="+error(result, yCV).toString)
//
//      // calculate training anc CV error
//      val intercept = DenseVector.ones[Double](x.rows)
//      val xAndIntercept = DenseMatrix.horzcat(intercept.toDenseMatrix.t, x)
//      val errorTrain = classifierLogit.error(xAndIntercept, DenseVector[Double](y.map(_.toDouble).toArray))
//
//      val interceptCV = DenseVector.ones[Double](xCV.rows)
//      val xCVAndIntercept = DenseMatrix.horzcat(interceptCV.toDenseMatrix.t, xCV)
//      val errorCV = classifierLogit.error(xCVAndIntercept, DenseVector[Double](result.map(_.toDouble).toArray))
//
//      (errorTrain, errorCV)
//    })

//    lambdas.zip(errors).foreach(println)
//    println("Training error:")
//    errors.map(_._1).foreach(x=>print(x+","))
//    println("\nCV error:")
//    errors.map(_._2).foreach(x=>print(x+","))
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
//    classifierKNN.useClusters = true

    val startTrainTime = System.nanoTime()
    classifierKNN.train(x, y)
    val trainTime = System.nanoTime()
    println("Training set length = "+classifierKNN.trainingLabels.length)
    println("Test set length = "+yCV.length)

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

    def printHist() = {
      val histTrain = y.groupBy(x => x).mapValues(_.size)
      println("Training set histogram: "+histTrain + "   Mean="+mean(histTrain.values.map(_.toDouble))+ "   STD="+stddev(histTrain.values.map(_.toDouble)))
      val histTest = yCV.groupBy(x => x).mapValues(_.size)
      println("Test set histogram: " + histTest + "   Mean=" + mean(histTest.values.map(_.toDouble)) + "   STD=" + stddev(histTest.values.map(_.toDouble)))
      val histPred = result.groupBy(x => x).mapValues(_.size)
      println("Prediction set histogram: " + histPred + "   Mean=" + mean(histPred.values.map(_.toDouble)) + "   STD=" + stddev(histPred.values.map(_.toDouble)))
    }

//    printHist()
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
