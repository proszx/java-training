
import java.util.concurrent.ExecutorService;// 提交借口
import java.util.concurrent.Executors;//执行借口
import java.util.concurrent.AbstractExecutorService;// 执行和提交接口 整合合并
import java.util.concurrent.ThreadPoolExecutor;//java普通线程池
import java.util.concurrent.TimeUnit;

public class JUC {
    public static void main(String[] args) {
        ExecutorService executorService=Executors.newCachedThreadPool();
        executorService.submit(new Runnable(){
        
            @Override
            public void run() {
                // TODO Auto-generated method stub
                System.out.println("这是一个测试案例");
            }
        });
        executorService.submit(()->System.out.println("test code"));
        executorService.execute(new Runnable(){
        
            @Override
            public void run() {
                System.out.println("这是一个测试案例");               
            }
        });
        //两者区别 submit execute
        //1.submit 又返回值 execute 没有返回值
        //execute  四种情况 是否小于核心 是 加入worker 开始执行
        //大于核心 第二部 workerQueue 把线程放入阻塞队列 poll take 获取
        //第三步 放入非核心 
        //第四部 拒绝
        //都 addWorker 工作队列 检测线程状态 CAS 状态和核心刷量
        // Worker 方法  AbstractQueuedSynchronizer  信号量 
        // 同时实现runnable run 方法内
        // runWorker 方法
        // gettask 方法级别的调用
        // 线程池的 processWorkerExit 重新返回 addWorker IO复用 
        //2.Future<T> submit
        //AbstractExecutorService 没有execute 执行
        //submit 是execute的封装
        //RunnableFuture
        executorService.shutdown();
        excute();
    }
    public static class SingletonFactory{
        private volatile static SingletonFactory INSTANCE;
        public static final SingletonFactory getInstance() {
            if(INSTANCE == null){
                synchronized(INSTANCE){
                    if(INSTANCE == null){
                        INSTANCE = new SingletonFactory();
                    }
                }
            }
            return INSTANCE;
        }
    }
        volatile static int a=0,b=0;
        static int x,y;
        public void execute() {
           int i=0;
           while (true){
               i++;
               x=0;
               y=0;
               a=0;
               b=0;
        Thread thread1=new Thread(new Runnable(){
    
                @Override
                public void run() {
                    // TODO Auto-generated method stub
                    //Thread.sleep(1000,TimeUnit.);
                    a=1;
                    UnSafe
                    x=b;
                }   
    
            });
        Thread thread2=new Thread(new Runnable(){
    
                @Override
                public void run() {
                    // TODO Auto-generated method stub
                   b=1;
                   y=a; 
                }
    
            });
            thread1.start();
            thread2.start();
           }
            
        }
}