import breeze.linalg.{DenseMatrix, DenseVector}
import scala.io.Source._
import scala.util.Random

class Loader {
  def loadIrisData(path: String) = {
    val dataInput = fromFile(path).getLines.map(_.split(",").map(_.trim))

    Random.setSeed(42)
    val data = Random.shuffle(dataInput).toArray
    val features = DenseMatrix((data.map(_(0).toDouble)),
                               (data.map(_(1).toDouble)),
                               (data.map(_(2).toDouble)),
                               (data.map(_(3).toDouble)) )
    // convert to binary classification task
    val labels = data.map(_(4)).map(name => if (name == "Iris-setosa") 1 else 0).toList

    (features.t, labels)
  }

  def loadMNISTData(path: String) = {
    val data = fromFile(path).getLines.map(_.split(",").map(_.toDouble) ).toArray
    val features = DenseMatrix(data.map(_.drop(1)): _*)
    val labels = data.map(_(0).toInt).toList
    (features, labels)
  }

  def loadMNISTDataBinary(path: String, num1: Int, num2: Int) = {
    val dataInput = fromFile(path).getLines.map(_.split(",").map(_.toDouble) ).toArray
    val data = dataInput.filter( ar => ar(0).toInt == num1 || ar(0).toInt == num2 )
    val features = DenseMatrix(data.map(_.drop(1)): _*)
    val labels = data.map( ar => if (ar(0).toInt == num1) 0 else 1 ).toList
    (features, labels)
  }
}
