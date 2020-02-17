import layer._
import pll.utils.RichArray._
object Addition{
  def loadData()={
    def fd(line:String) = line.split("")
    val file = scala.io.Source.fromFile("/home/share/addition.txt").getLines.map(fd).toArray
    val data = file.map(makeData)//data.size=5000
    data
  }//loadData

  def makeData(file:Array[String])={
    //file.size:12:データ一行における数式の文字の個数
    val target = "0123456789+_ "
    var data = Array.ofDim[Float](file.size,target.size)
    //onehotVevtor生成
    for(i<-0 until file.size){
      data(i)(target.indexOf(file(i))) = 1
    }
    data
  }//makeData


  def main(){
    def makeString(xs:Array[Array[Float]])={
      val target = "0123456789+_ "
      var s = ""
      for(i<-0 until xs.size){
        val index = xs(i).indexOf(xs(i).max)
        s += target(index)
      }
      s
    }

    def subArray(a:Array[Float],b:Array[Float])={
      var c = new Array[Float](a.size)
      for(i<-0 until a.size){
        c(i) = a(i) - b(i)
      }
      c
    }

    val ln = 100
    //------network--------------
    val N = new network()
    val enLS = new LSTM(100,100)
    val deLS = new LSTM(100,100)
    val encode = List(
      new Affine(13,100),
      enLS
    )
   val decode = List(
     new Affine(13,100),
     deLS,
     new Affine(100,13),
     new Softmax()
   )

    //------data-----------------
    //answer
    val file = scala.io.Source.fromFile("/home/share/addition.txt").getLines.toArray
    //input
    val data = loadData()
    //------learning--------------
    for(i<-0 until ln){
     
      for(p<-0 until 1){
        print(i+":answer-->"+file(p).takeRight(5))
        //----forward----
        //encode
        for(q<-0 until 7){
          N.forwards(encode,data(p)(q))
        }
        //decode<-encode
        deLS.hs = enLS.hs
        //decode
        var Zs = Array.ofDim[Float](data.size,4,13)
        var count = 0
        for(q<-7 until data(0).size-1){
          Zs(p)(count) = N.forwards(decode,data(p)(q))
          count += 1
        }
        //resultPrint
        println(i+":result-->"+makeString(Zs(p)))
        //----backward----
        //decode
        var count2 = 0
        for(q<-7 until data(0).size-1){
          N.backwards(decode.reverse,subArray(Zs(p)(count2),data(p)(q+1)))
          count2 += 1
        }
        //encode<-decode
        enLS.dr = deLS.dr
        //encode
        val zero = Array.ofDim[Float](data.size,7,100)
        for(q<-0 until 7){
          N.backwards(encode.reverse,zero(p)(q))
        }
        N.updates(encode)
        N.updates(decode)
        N.resets(encode)
        N.resets(decode)

      }//datasize
    }//ln

  }//main
}//object
