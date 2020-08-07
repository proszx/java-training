package com.company;

import com.company.aspj.Service;
import com.company.aspj.ServiceImpl;
import com.company.aspj.User;
import com.company.juc.aqs.CountdownLatchSample;
import com.company.juc.aqs.SemaphoreExample;
import com.company.juc.blockingQueue.ProducerConsumer;
import com.company.juc.fork.ForkJoinExample;
import com.company.juc.futureTask.FutureTaskExample;
import com.company.juc.join.JoinExmaple;
import com.company.juc.lock.LockSample;
import com.company.juc.thread.InterruptedSample;
import com.company.juc.thread.MyCallable;
import com.company.juc.thread.MyThead;
import com.company.juc.thread.extentThread;
import com.company.juc.threadpool.SyncExamples;
import com.company.juc.unsafe.ThreadUnsafeExample;
import com.company.juc.waitnotify.AwaitSignalSample;
import com.company.juc.waitnotify.WaitNotifyExample;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class Main {

    public  static  class TestCode{
        public  static  void testThread() throws ExecutionException, InterruptedException {
            // write your code here
            MyThead myThead=new MyThead();
            myThead.start();

            extentThread eThread=new extentThread();
            Thread thread=new Thread(eThread);
            thread.setDaemon(true);
            thread.start();

            MyCallable mc=new MyCallable();
            FutureTask<Integer> ft=new FutureTask<>(mc);
            Thread threads=new Thread(ft);
            threads.start();
            System.out.println(ft.get());

            ExecutorService executorService= Executors.newCachedThreadPool();
            for(int i=0;i<5;i++){
                executorService.execute(new extentThread());
            }
            executorService.shutdown();

            Thread thread1=new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        Thread.sleep(2000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            });

            Thread thread2=new Thread(new Runnable() {
                @Override
                public void run() {
                    //线程切换
                    Thread.yield();
                }
            });

            InterruptedSample nSample=new InterruptedSample();
            nSample.start();
        }
        public static  void testThreadPool(){
            ExecutorService executorService=Executors.newCachedThreadPool();
            executorService.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        Thread.sleep(1000);
                    }catch (InterruptedException e){
                        e.printStackTrace();
                    }
                }
            });
            System.out.println("finish thread main");
            executorService.shutdownNow();

            Future future=executorService.submit(new Runnable() {
                @Override
                public void run() {
                    System.out.println("begin stop");
                }
            });
            future.cancel(true);

            SyncExamples syncExamples=new SyncExamples();
            SyncExamples examples=new SyncExamples();
            ExecutorService executorService1=Executors.newCachedThreadPool();
            executorService1.submit(()->syncExamples.lockClazz());
            System.out.println("dd");
            executorService1.submit(()->syncExamples.lockFunc());
            System.out.println("dd");
            executorService1.submit(()->syncExamples.lockBlock());
            System.out.println("dd");
            executorService1.submit(()->examples.lockStatic());
            System.out.println("dd");
            executorService1.shutdown();
        }
        static void testLock(){
            LockSample lockSample=new LockSample();
            lockSample.Lock();
        }
        static  void testJoin(){
            JoinExmaple joinExmaple=new JoinExmaple();
            joinExmaple.run();
        }
        static  void testWaitNoti() throws InterruptedException {
            WaitNotifyExample waitNotifyExample=new WaitNotifyExample();
            waitNotifyExample.run();

            System.out.println("Condition");
            AwaitSignalSample awaitSignalSample=new AwaitSignalSample();
            awaitSignalSample.run();

        }
        static  void testCountDownLatch() throws InterruptedException {
            CountdownLatchSample countdownLatchSample=new CountdownLatchSample();
            countdownLatchSample.CylicBarrierrun();
            countdownLatchSample.run();
        }
        static  void testSemaphore(){
            SemaphoreExample semaphoreExample=new SemaphoreExample();
            semaphoreExample.run();
        }
        static  void testFuture() throws ExecutionException, InterruptedException {
            FutureTaskExample futureTaskExample=new FutureTaskExample();
            FutureTask t=futureTaskExample.run();
            Thread th=new Thread(t);
            th.start();
            System.out.println(t.get());
        }
        static  void testPC(){
            ProducerConsumer producerConsumer=new ProducerConsumer();
            producerConsumer.run();
        }
        static  void  testParall() throws ExecutionException, InterruptedException {
            ForkJoinExample forkJoinExample=new ForkJoinExample(1,21483648);
            ForkJoinPool forkJoinPool=new ForkJoinPool();
            Future ft=forkJoinPool.submit(forkJoinExample);

            System.out.print(ft.get());

        }
        static  void testUnsafe() throws InterruptedException {
            ExecutorService executorService=Executors.newCachedThreadPool();
            ThreadUnsafeExample threadUnsafeExample=new ThreadUnsafeExample();
            final  CountDownLatch countDownLatch=new CountDownLatch(1000);
            for(int i=0;i<1000;i++){
                executorService.submit(()->{
                    threadUnsafeExample.setCnt();
                    threadUnsafeExample.setCount();
                    threadUnsafeExample.setSyncCount();
                    threadUnsafeExample.setAtoCnt();

                    countDownLatch.countDown();
                });
            }
            countDownLatch.await();
            //这里必须配合使用程序计数器
            executorService.shutdown();
            System.out.println(threadUnsafeExample.getCnt());
            System.out.println(threadUnsafeExample.getCount());
            System.out.println(threadUnsafeExample.getSyncCount());
            System.out.println(threadUnsafeExample.getAtoCnt());
        }
    }
    private static  void aspj(){
        ServiceImpl service=new ServiceImpl();
        User user=new User(123,"prozx");

        service.printUser(user);
    }
    private static  void dateTest() throws ParseException {
        SimpleDateFormat format=new SimpleDateFormat("yyyy-MM-dd HH:mm:mm");
        Date date=format.parse("2020-07-29 01:57:00");
        System.out.println(date);



    }
    /**
     * 给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。
     * 你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。
     * 1+2*3 => (1+2)*3
     * @param input
     */
    private static HashMap<String,List<Integer>> map=new HashMap<>();
    public static List<Integer> diffWaysToCompute(String input) {
        if (map.containsKey(input))
            return map.get(input);
        List<Integer> list = new ArrayList<>();
        int len = input.length();
        for (int i = 0; i < len; i++) {
            char c = input.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                List<Integer> left = diffWaysToCompute(input.substring(0, i));
                List<Integer> right = diffWaysToCompute(input.substring(i + 1, len));
                for (int l : left) {
                    for (int r : right) {
                        switch (c) {
                            case '+':
                                list.add(l + r);
                                break;
                            case '-':
                                list.add(l - r);
                                break;
                            case '*':
                                list.add(l * r);
                                break;
                        }
                    }
                }
            }
        }
        if (list.size() == 0)
            list.add(Integer.valueOf(input));
        map.put(input, list);
        return list;
    }
     /**
         * 给定一个字符串 s，你可以通过在字符串前面
         * 添加字符将其转换为回文串。
         * 找到并返回可以用这种方式转换的最短回文串。
         * @param s
         * @return
         */
        public String shortestPalindrome(String s) {
            if(s==null){
                return null;
            }
            StringBuffer sb=new StringBuffer(s).reverse();
            String str=s+"#"+sb;
            int[] next=next(str);
            String prefix=sb.substring(0,sb.length()-next[sb.length()]);
            return prefix+s;

        }
        private int[] next(String str){
            int[] tmp=new int[str.length()+1];
            int k=-1;
            tmp[0]=-1;
            int i=1;
            while(i<str.length()){
                if(k==-1||str.charAt(k)==str.charAt(i-1)){
                    tmp[i++]=++k;
                }else{
                    tmp[k]=k;
                }
            }
            return tmp;
        }
    public static void main(String[] args) throws ExecutionException, InterruptedException, ParseException {

        // TestCode.testThread();
        // TestCode.testThreadPool();
        // TestCode.testJoin();
        // TestCode.testCountDownLatch();
        // TestCode.testSemaphore();
        // TestCode.testFuture();
        // TestCode.testPC();
        // TestCode.testParall();
        // TestCode.testUnsafe();
        // long num=Long.MAX_VALUE;
        // int cnt=0;
        // while(num!=0){
        // num/=2;
        // cnt++;
        // }
        // int nums=Integer.MIN_VALUE;
        //
        // System.out.println(cnt);
        // aspj();
        // System.out.print(System.getProperty("os.name"));
        //System.out.println(diffWaysToCompute("1+2*3-4+20*3"));
        //System.out.println(Double.parseDouble("24e-3"));
            Integer i=1;
            test(i);
            System.out.print(i);
    }
    private  static  Integer test(Integer i) {
        i += 1;
        return i < 8888888 ? test(i) : i;
    }
}
