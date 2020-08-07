//package com.company.juc.jucdesign.eventbus;
//
//import java.lang.reflect.Method;
//import java.util.concurrent.ConcurrentLinkedQueue;
//import java.util.concurrent.Executor;
//import java.util.concurrent.ExecutorService;
//
//
//
//public class Dispatcher {
//    private final Executor executor;
//    private final EventExceptionHandler exceptionHandler;
//    public static final Executor SEQ_EXECUTER_SERVICE=SeqExecuterService.INSTANCE;
//    public static final Executor PRE_THREAD_EXECUTOR=PreThreadExecutor.INSTANCE;
//
//    private Dispatcher(Executor executor,EventExceptionHandler exceptionHandler){
//        this.executor=executor;
//        this.exceptionHandler=exceptionHandler;
//    }
//    public void dispatch(Bus bus,Register register,Object event,String topic){
//        ConcurrentLinkedQueue<Subscriber> subscriber=register.scanSubscriber(topic);
//        if(null==subscriber){
//            if(exceptionHandler!=null){
//                exceptionHandler.handle(new IllegalArgumentException("The topic is not bind yet"),new BaseEventContext(bus.getBusName(),null,event));
//
//            }
//            return;
//        }
//        subscriber.stream()
//        .filter(subscriber->!subscriber.isDisable())
//        .filter(subscriber->{
//            Method subsMethod=subscriber.getSubscribeMethods();
//            Class<?> a=subscriber.getParameterTypes()[0];
//            return (a.isAssignableFrom(event.getClass()));
//        }).forEach(subscriber->realInvokeSubscribe(subscriber,event,bus));
//    }
//
//    private void realInvokeSubscribe(Subscriber subscriber,Object event,Bus bus){
//        Method subsMethod=subscriber.getSubscribeMethods();
//        Object subsObject=subscriber.getSubscribeObject();
//        executor.execute(()->{
//            try {
//                subsMethod.invoke(subsObject, event);
//            } catch (Exception e) {
//                //TODO: handle exception
//                if(null!=exceptionHandler){
//                    exceptionHandler.handle(e,new BaseEventContext(bus.getBusName(),subscriber,event));
//                }
//            }
//        });
//    }
//    public void close(){
//        if(executor instanceof ExecutorService){
//            ((ExecutorService) executor).shutdown();
//        }
//    }
//    static Dispatcher newDispatcher(EventExceptionHandler exceptionHandler,Executor executor){
//        return new Dispatcher(executor,exceptionHandler);
//    }
//    static Dispatcher seqDispatcher(EventExceptionHandler exceptionHandler){
//        return  new Dispatcher(SEQ_EXECUTER_SERVICE, exceptionHandler);
//    }
//    static Dispatcher preDispatcher(EventExceptionHandler exceptionHandler){
//        return new Dispatcher(PRE_THREAD_EXECUTOR, exceptionHandler);
//    }
//    private static class SeqExecuterService implements Executor{
//        private final static SeqExecuterService INSTANCE=new SeqExecuterService();
//        @Override
//        public void execute(Runnable cmd) {
//            cmd.run();
//        }
//    }
//    private static class PreThreadExecutor implements Executor{
//        private final static PreThreadExecutor INSTANCE=new PreThreadExecutor();
//        @Override
//        public void execute(Runnable mcd){
//            new Thread(mcd).start();
//        }
//
//    }
//    private static class BaseEventContext implements EventContext{
//        private final String eventBusName;
//        private final Subscriber subscriber;
//        private final Object event;
//
//        private BaseEventContext(String eventBusName,Subscriber subscriber,Object event){
//            this.event=event;
//            this.eventBusName=eventBusName;
//            this.subscriber=subscriber;
//        }
//
//		@Override
//		public String getSource() {
//			// TODO Auto-generated method stub
//			return this.eventBusName;
//		}
//
//		@Override
//		public Object getSubcriber() {
//			// TODO Auto-generated method stub
//			return subscriber!=null ? subscriber.getSubscribeObject():null;
//		}
//
//		@Override
//		public Method getSubcribe() {
//			// TODO Auto-generated method stub
//			return subscriber!=null ? subscriber.getSubscribeMethods():null;
//
//        }
//        @Override
//        public Object getEvent(){
//            return this.event;
//        }
//
//
//    }
//}
