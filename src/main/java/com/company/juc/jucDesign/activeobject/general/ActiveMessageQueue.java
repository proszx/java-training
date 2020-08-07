package com.company.juc.jucdesign.activeobject.general;
import java.util.LinkedList;

public class ActiveMessageQueue {
    private final LinkedList<ActiveMessage> msg=new LinkedList<>();
    public ActiveMessageQueue(){
        new ActiveDaemonThread(this).start();
    }
    public void offer(ActiveMessage activeMsg){
        synchronized(this){
            msg.addLast(activeMsg);
            this.notify();
        }
    }
    
    public synchronized ActiveMessage take(){
        while(msg.isEmpty()){
            try {
                this.wait();
            } catch (Exception e) {
                //TODO: handle exception
                e.printStackTrace();
            }
        }
        return msg.removeFirst();
    }
}