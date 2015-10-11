import breeze.linalg.{DenseVector, DenseMatrix}

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
  override def predict(data: DenseMatrix[Double]): DenseVector[Int] = DenseVector.zeros[Int](data.size)
}