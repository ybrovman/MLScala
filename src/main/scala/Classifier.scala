import breeze.linalg.{*, Axis, DenseMatrix, sum}
import breeze.numerics.sqrt

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

  override def train(data: DenseMatrix[Double], labels: List[Int]) = {
    require(data.rows == labels.length, s"data (length = ${data.rows}) should have same size as labels (length = ${labels.length})")
    trainingData = data
    trainingLabels = labels
  }

  override def predict(data: DenseMatrix[Double]): List[Int] = {
    require(data.cols == trainingData.cols, s"prediction data (cols = ${data.cols}) should have same size as training data (cols = ${trainingData.cols})")

    val allDist = predictDist(data)
    val predictions = getKNNFromDist(allDist)

    predictions
  }

  def predictDist(data: DenseMatrix[Double]): List[List[(Double, Int)]] = {
    val numExamples = data.rows
    val examples = List.range(0, numExamples)

    val predictions = examples.map( x => {
      val vec1 = data(x, ::).t
      val diff = trainingData(*, ::) - vec1
      val dist = sqrt(sum(diff :* diff, Axis._1))
      val distSorted = dist.toArray.toList.zip(trainingLabels).sorted
      distSorted
    })

    predictions
  }

  def getModes(inputList: List[Int]): List[Int] = {
    val grouped = inputList.groupBy(x => x).mapValues(_.size)
    val modeValue = grouped.maxBy(_._2)._2
    val modes = grouped.filter(_._2 == modeValue).map(_._1).toList
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