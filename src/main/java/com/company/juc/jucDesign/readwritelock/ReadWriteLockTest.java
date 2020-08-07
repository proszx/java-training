package readwritelock;
public class ReadWriteLockTest {
    private final static String text="templateforreadding";
    public static void main(String[] args) {
        final ShareData shareData=new ShareData(50);
        for(int i=0;i<2;i++){
            new Thread(()->{
                for(int index=0;index<text.length();index++){
                    try {
                        char c=text.charAt(index);
                        shareData.write(c);
                        System.out.println(Thread.currentThread()+"------ write ------"+c);
                    } catch (InterruptedException e) {
                        //TODO: handle exception
                        e.printStackTrace();
                    }
                }
            }).start();
        }
        for(int i=0;i<10;i++){
            new Thread(()->{
                while(true){
                    try {
                        
                        String temp=new String(shareData.read());
                        System.out.println(Thread.currentThread()+"------ read ------"+temp);
                    } catch (InterruptedException e) {
                        //TODO: handle exception
                        e.printStackTrace();
                    }
                }
            }).start();
        }
    }
}