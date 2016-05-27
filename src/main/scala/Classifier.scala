import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.{abs, exp, round, sqrt}

trait Classifier {
  def train(data: DenseMatrix[Double], lables: List[Int])
  def predict(data: DenseMatrix[Double]): List[Int]
}

class LogistricRegression() extends Classifier {
  var beta: DenseVector[Double] = null

  override def train(data: DenseMatrix[Double], labels: List[Int]) = {
    require(data.rows == labels.length, s"data (length = ${data.rows}) should have same size as labels (length = ${labels.length})")
    val M = data.rows

    beta = DenseVector.ones[Double](data.cols + 1) // initialize beta to all 1's
    val alpha = 1.0 // learning rate
    val lambda = 1.0 // regularization parameter
    val convergenceTolerance = 0.001

    val intercept = DenseVector.ones[Double](data.rows)
    val dataAndIntercept = DenseMatrix.horzcat(intercept.toDenseMatrix.t, data)
    val y = DenseVector[Double](labels.map(_.toDouble).toArray)

    // need to recursively find all beta values
    val coef = alpha / M
    List.range(0,8).map( x => {
      val prod = coef * dataAndIntercept.t * (sigmoid(dataAndIntercept * beta) - y)

      val betaInterceptTemp = beta(0) - prod(0)
      beta = beta * (1 - coef * lambda) - prod
      beta(0) = betaInterceptTemp

//      println("iteration "+x+"   "+beta)
    })
  }

  override def predict(data: DenseMatrix[Double]): List[Int] = {
    require(data.cols+1 == beta.length, s"data (columns = ${data.cols}) + 1 should have same size as beta (length = ${beta.length})")
    val intercept = DenseVector.ones[Double](data.rows)
    val dataAndIntercept = DenseMatrix.horzcat(intercept.toDenseMatrix.t, data)

    val predictions = round(sigmoid(dataAndIntercept * beta)).toArray.toList.map(_.toInt)

    predictions
  }

  def sigmoid(x: DenseVector[Double]): DenseVector[Double] = {
    val ones = DenseVector.ones[Double](x.length)
    ones / (ones + exp(-x))
  }
}

class KNN() extends Classifier {
  var trainingData: DenseMatrix[Double] = null
  var trainingLabels: List[Int] = null
  var k: Int = 1 // default to 1-nearest neighbor

  var useClusters = false
  var trainingClusterMeans: DenseMatrix[Double] = null
  var trainingClusterLabels: List[Int] = null

  override def train(data: DenseMatrix[Double], labels: List[Int]) = {
    require(data.rows == labels.length, s"data (length = ${data.rows}) should have same size as labels (length = ${labels.length})")
    trainingData = data
    trainingLabels = labels

    if (useClusters) calculateClusterMeans()
  }

  override def predict(data: DenseMatrix[Double]): List[Int] = {
    require(data.cols == trainingData.cols, s"prediction data (cols = ${data.cols}) should have same size as training data (cols = ${trainingData.cols})")

    val allDist = predictDist(data)
    val predictions = getKNNFromDist(allDist)

    predictions
  }

  def calculateClusterMeans() = {
    trainingClusterLabels = trainingLabels.toSet[Int].toList
    println(trainingClusterLabels)
    val numClusters = trainingClusterLabels.size
    val cols = trainingData.cols
    var arrayMeans = Array[DenseVector[Double]]()

    trainingClusterLabels.map( label => {
      var tempSum = DenseVector.zeros[Double](cols)
      val labelIndex = trainingLabels.zipWithIndex.filter(_._1 == label).map(_._2)

      labelIndex.map( index => tempSum = tempSum + trainingData(index, ::).t )
      val mean = tempSum :/ numClusters.toDouble
      arrayMeans :+= mean
    })

    trainingClusterMeans = DenseMatrix(arrayMeans.map(_.toArray): _*)
  }

  def predictDist(data: DenseMatrix[Double], distance: (DenseVector[Double]) => DenseVector[Double] = distanceEuclidean): List[List[(Double, Int)]] = {
    val numExamples = data.rows
    val examples = List.range(0, numExamples)

    val predictions = examples.map( x => {
      val vec = data(x, ::).t
      val dist = distance(vec)
      val labels = if (useClusters) trainingClusterLabels else trainingLabels
      val distSorted = dist.toArray.toList.zip(labels).sorted
      distSorted
    })

    predictions
  }

  def distanceEuclidean(inputVector: DenseVector[Double]): DenseVector[Double] = {
    val diff = if (useClusters) trainingClusterMeans(*, ::) - inputVector else trainingData(*, ::) - inputVector
    val dist = sqrt(sum(diff :* diff, Axis._1))
    dist
  }

  def distanceManhattan(inputVector: DenseVector[Double]): DenseVector[Double] = {
    val diff = if (useClusters) trainingClusterMeans(*, ::) - inputVector else trainingData(*, ::) - inputVector
    val dist = sum(abs(diff), Axis._1)
    dist
  }

  def getModes(inputList: List[Int]): List[Int] = {
    val grouped = inputList.groupBy(x => x).mapValues(_.size)
    val modeValue = grouped.maxBy(_._2)._2
    val modes = grouped.filter(_._2 == modeValue).keys.toList
    modes
  }

  def getKNNFromDist(allDist: List[List[(Double,Int)]]): List[Int] = {
    allDist.map( x => {
      val distSortedLabel = x.map(_._2)

      var tempK = k
      var modes = getModes(distSortedLabel.take(tempK))
      val predictionLabel = {
        while (modes.size > 1) {
          tempK -= 1 // reduce tempK by 1 if there are more than 1 majority classes
          modes = getModes(distSortedLabel.take(tempK))
        }
        modes.head
      }
      predictionLabel
    })
  }
}