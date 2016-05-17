import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.{abs, sqrt}

trait Classifier {
  def train(data: DenseMatrix[Double], lables: List[Int])
  def predict(data: DenseMatrix[Double]): List[Int]
}

class LogistricRegression() extends Classifier {
  override def train(data: DenseMatrix[Double], labels: List[Int]) = println(data.size)
  override def predict(data: DenseMatrix[Double]): List[Int] = List.fill(data.size)(0)
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