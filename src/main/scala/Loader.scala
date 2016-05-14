import breeze.linalg.{DenseMatrix, DenseVector}
import scala.io.Source._

class Loader {
  def loadIrisData(path: String) = {
    val data = fromFile(path).getLines.map(_.split(",").map(_.trim)).toArray
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
}
