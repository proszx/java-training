package com.company.juc.jucdesign.activeobject;

import java.util.LinkedList;

import com.company.juc.jucdesign.activeobject.general.ActiveMessage;

public class ActiveMessageQueue {

    private final LinkedList<MethodMessage> msg=new LinkedList<>();
    public ActiveMessageQueue(){
        new ActiveDaemonThread(this).start();
    }
	public void offer(MethodMessage methodMessage) {
        synchronized(this){
            msg.addLast(methodMessage);
            this.notify();
        }
    }

    public synchronized MethodMessage take(){
        while(msg.isEmpty()){
            try {
                this.wait();
            } catch (InterruptedException e) {
                //TODO: handle exception
                e.printStackTrace();
            }
        }
        return msg.removeFirst();
        
    }
	public void offer(ActiveMessage build) {
	}

}
