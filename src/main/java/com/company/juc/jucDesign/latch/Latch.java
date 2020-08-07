package latch;

import java.util.concurrent.TimeUnit;

/** 
 * 等待开闸 不管先来后到
 * 等到一定批次 才放行
 * 
*/
public  abstract class Latch {
    protected int limit;
    public Latch(int limit){
        this.limit=limit;
    }
    public abstract void await() throws InterruptedException;

    public abstract void await(TimeUnit unit,long time) throws InterruptedException,WaitTimeException;
    public abstract void countDown();

    public abstract int getUnarrived();
}