package threadpermsg;

import java.util.concurrent.TimeUnit;

public class TaskHandler implements Runnable{
    private final Request req;
    public TaskHandler(Request req){
        this.req=req;
    }

	@Override
	public void run() {
        // TODO Auto-generated method stub
        System.out.println("begin handleer"+req);
        slowly();
        System.out.println("end handler"+req);
		
    }
    public void slowly(){
        try {
			TimeUnit.SECONDS.sleep(1);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

}