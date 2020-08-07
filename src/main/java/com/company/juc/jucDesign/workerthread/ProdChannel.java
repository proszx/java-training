package workerthread;

public class ProdChannel {
    private final static int MAX_PROD=100;
    private final Production[] prodQueue;
    private int tail;
    private int head;
    private int total;
    private final Worker[] worker;
    public ProdChannel(int workerSize){
        this.worker=new Worker[workerSize];
        this.prodQueue=new Production[MAX_PROD];
        for(int i=0;i<workerSize;i++){
            worker[i]=new Worker("worker-"+i,this);
            worker[i].start();
        }
        
    }
    public void offerProduction(Production prod){
        synchronized(this){
            while(total>=prodQueue.length){
                try {
                    this.wait();
                } catch (InterruptedException e) {
                    //TODO: handle exception
                    e.printStackTrace();
                }
            }
                prodQueue[tail]=prod;
                tail=(tail+1)%prodQueue.length;
                total++;
                this.notifyAll();
        }
    }
    public Production takeProd(){
        synchronized(this){
            while(total<=0){
                try {
                    this.wait();
                } catch (InterruptedException e) {
                    //TODO: handle exception
                    e.printStackTrace();
                }
            }
            Production prods=prodQueue[head];
            head=(head+1)%prodQueue.length;
            total--;
            this.notifyAll();
            return prods;
        }
    }
}