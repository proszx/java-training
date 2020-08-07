package com.company.juc.join;

public class JoinExmaple {
    private class ThreadA extends  Thread{
        @Override
        public void run() {
            System.out.println("A is running");

        }
    }
    private class ThreadB extends Thread{
        ThreadA threadA;
        ThreadB(ThreadA a){
            this.threadA=a;
        }

        @Override
        public void run() {
            try {
                //线程B会等待线程A
                threadA.join();
            }catch (InterruptedException e){
                e.printStackTrace();
            }
            System.out.println("ThreadB running");
        }
    }
    public  void run(){
        ThreadA A=new ThreadA();
        ThreadB B=new ThreadB(A);
        B.start();
        //A.start();
    }
}
