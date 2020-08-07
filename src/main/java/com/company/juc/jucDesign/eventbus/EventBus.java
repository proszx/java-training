//package com.company.juc.jucdesign.eventbus;
//
//public class EventBus implements Bus{
//    private final Register regi=new Register();
//    private String busName;
//    private final static String DEFAULT_NAME="default";
//    private final static String DEFAULT_TOPIC="default-topic";
//    private final Dispatcher dispatcher;
//    public EventBus(){
//        this(DEFAULT_NAME,null,Dispatcher.SEQ_EXECUTER_SERVICE);
//    }
//    public EventBus(String busName){
//        this(busName,null,Dispatcher.SEQ_EXECUTER_SERVICE);
//    }
//    public EventBus(String busName,EventExceptionHandler exceptionHandler,Executor executor){
//        this.busName=busName;
//        this.dispather=Dispatcher.newDispatcher(exceptionHandler,executor);
//    }
//    public EventBus(EventExceptionHandler exceptionHandler){
//       this(DEFAULT_NAME,exceptionHandler,Dispatcher.SEQ_EXECUTER_SERVICE);
//    }
//    @Override
//    public void register(Object subscriber) {
//        // TODO Auto-generated method stub
//        this.regi.bind(subscriber);
//
//    }
//
//    @Override
//    public void unregister(Object subscriber) {
//        // TODO Auto-generated method stub
//        this.regi.unbind(subscriber);
//    }
//
//    @Override
//    public void post(Object event) {
//        // TODO Auto-generated method stub
//        this.post(event, DEFAULT_TOPIC);
//
//    }
//
//    @Override
//    public void post(Object event, String topic) {
//        // TODO Auto-generated method stub
//        //this.post(event,topic);
//        this.dispatcher.dispatch(this,regi,event,topic);
//    }
//
//    @Override
//    public void close() {
//        // TODO Auto-generated method stub
//        this.dispatcher.close();
//
//    }
//
//    @Override
//    public String getBusName() {
//        // TODO Auto-generated method stub
//        return this.busName;
//    }
//}