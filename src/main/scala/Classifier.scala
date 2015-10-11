import breeze.linalg.{DenseVector, DenseMatrix}

trait Classifier {
  def train(data: DenseMatrix[Double], lables: DenseVector[Double])
  def predict(data: DenseMatrix[Double]): DenseVector[Double]
}

class LogistricRegression() extends Classifier {
  override def train(data: DenseMatrix[Double], lables: DenseVector[Double]) = println(data.size)
  override def predict(data: DenseMatrix[Double]): DenseVector[Double] = DenseVector.zeros[Double](data.size)
}