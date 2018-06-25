package Alogrithm.OOD;

//public class Singleton {
//    private static Singleton s = null;
//    static class SingletonInstance{
//        static Singleton singleton = new Singleton();
//    }
//    private Singleton(){}
//    public static Singleton getInstance(){
//        return SingletonInstance.singleton;
//    }
//
//
//    public static void main(String[] args){
//        TreeMap<Integer,Integer> map = new TreeMap<>();
//    }


//
//}

/*
public class Singleton{
static class SingletonInstance{
static Singleton singleton = new Singleton();
}
private Singleton(){}
public static Singleton getInstance(){
return SingletonInstance.singleton;}
}
 */

class Singleton {
    /**
     * @return: The same instance of this class every time
     */
    public static Singleton instance = null;
    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
    enum type{
        a,b,c,d;
    }
    class inner{
        type p;

    }
    public static void main(String[] args){
        type p = type.a;

    }
}


