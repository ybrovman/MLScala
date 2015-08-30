import breeze.linalg._

object classifier {
  def main(args: Array[String]) {
    println("Training the classifier...")
    val x = DenseVector.zeros[Double](5)
    val m = DenseMatrix.zeros[Int](5,5)
    m(4,::) := DenseVector(1,2,3,4,5).t


    val v1 = DenseVector[Double](1,2,3,4,5)
    val v2 = v1 :* DenseVector.fill(5){2.0}
    val v3 = DenseVector[Double](1,2,1,1,1)
    println(v2.t * v1)
    println(v1 + v3)
  }
}
