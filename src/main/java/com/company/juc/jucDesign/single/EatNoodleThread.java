package single;
public class EatNoodleThread extends Thread{
    private final String name;
    private final Tableware rightTableware;
    private final Tableware leftTableware;
    
    public EatNoodleThread(String name, Tableware rightTableware, Tableware leftTableware) {
        this.name=name;
        this.leftTableware=leftTableware;
        this.rightTableware=rightTableware;
    }

    @Override
    public void run(){
        while(true){
            this.eat();
        }
    }
    private void eat(){
        synchronized(leftTableware){
            StringBuffer sb=new StringBuffer();
            sb.append(name).append("take up").append(leftTableware).append("left");
            System.out.println(sb.toString());
            synchronized(rightTableware){
                StringBuffer sb2=new StringBuffer();
                sb2.append(name).append("take up").append(rightTableware).append("right");
                System.out.println(sb2.toString());
                System.out.println("eating now!!!");
                System.out.println(name+"put down"+rightTableware+" right");
            }
            System.out.println(name+"put down"+leftTableware+" left");
        }
    }
    public static void main(String[] args) {
        Tableware fork=new Tableware("fork");
        Tableware knife=new Tableware("knife");
        new EatNoodleThread("A", knife,fork).start();
        new EatNoodleThread("B",fork,knife).start();

    }
}