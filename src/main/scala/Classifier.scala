import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.sqrt
import scala.util.Sorting

trait Classifier {
  def train(data: DenseMatrix[Double], lables: DenseVector[Int])
  def predict(data: DenseMatrix[Double]): DenseVector[Int]
}

class LogistricRegression() extends Classifier {
  override def train(data: DenseMatrix[Double], labels: DenseVector[Int]) = println(data.size)
  override def predict(data: DenseMatrix[Double]): DenseVector[Int] = DenseVector.zeros[Int](data.size)
}

class KNN() extends Classifier {
  var trainingData: DenseMatrix[Double] = null
  var trainingLabels: DenseVector[Int] = null

  override def train(data: DenseMatrix[Double], labels: DenseVector[Int]) = {
    require(data.rows == labels.length, s"data (length = ${data.rows}) should have same size as labels (length = ${labels.length})")
    trainingData = data
    trainingLabels = labels
  }

  override def predict(data: DenseMatrix[Double]): DenseVector[Int] = {
    require(data.cols == trainingData.cols, s"prediction data (cols = ${data.cols}) should have same size as training data (cols = ${trainingData.cols})")
    val numExamples = data.rows
    val x = List.range(0, numExamples)
    val trainingLabelsList = trainingLabels.toArray.toList

    val vec1 = data(0, ::).t
    val diff = trainingData(*, ::) - vec1
    val dist = sqrt( sum( diff :* diff, Axis._1) )
    val distSorted = dist.toArray.toList.zip(trainingLabelsList).sorted

    println(dist.length, distSorted)

    println(dist.toArray.toList.slice(0,10).length, dist.toArray.toList.slice(0,10).toSet.size)


    DenseVector.zeros[Int](data.size)
  }
}