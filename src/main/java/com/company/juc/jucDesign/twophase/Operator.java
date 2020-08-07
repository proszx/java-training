package twophase;
import java.util.concurrent.ThreadPoolExecutor;

public class Operator {
    public void call(StringBuffer bus){
        TaskHandler taskHandler=new TaskHandler(new Request(bus));
        new Thread(taskHandler).start();
    }
    
    public void callback(StringBuffer bus){
        
        TaskHandler taskHandler=new TaskHandler(new Request(bus));
        ThreadPoolExecutor threadPool=new ThreadPoolExecutor(2, 6, 0, null, null);
        threadPool.execute(taskHandler);
    }
}