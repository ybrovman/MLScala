import breeze.linalg.{DenseVector, DenseMatrix}

trait Classifier {
  def train(data: DenseMatrix[Double], lables: DenseVector[Int])
  def predict(data: DenseMatrix[Double]): DenseVector[Int]
}

class LogistricRegression() extends Classifier {
  override def train(data: DenseMatrix[Double], lables: DenseVector[Int]) = println(data.size)
  override def predict(data: DenseMatrix[Double]): DenseVector[Int] = DenseVector.zeros[Int](data.size)
}