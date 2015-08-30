import breeze.linalg._

object classifier {
  def main(args: Array[String]) {
    println("Training the classifier...")
    val path = "data/iris.data"
    val myLoader = new Loader
    val (features,labels) = myLoader.loadFile(path)
    println(features.rows)
//    labels.foreach(println)

  }
}
