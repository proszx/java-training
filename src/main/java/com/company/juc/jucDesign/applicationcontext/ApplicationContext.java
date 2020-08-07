package com.company.juc.jucdesign.applicationcontext;
import java.util.concurrent.ConcurrentHashMap;

public final class ApplicationContext {
    /**private ApplicationConfiguration applicationConfiguration;
    private RuntimeInfo runtimeInfo;
    private static class Holder{
        private static ApplicationContext instance=new ApplicationContext();
        
        public static ApplicationContext getContext(){
            return Holder.instance;
        }

        public void setConfiguration(ApplicationConfiguration applicationConfiguration){
            this.applicationConfiguration=applicationConfiguration;
        }
        public ApplicationConfiguration getConfiguration(){
            return this.applicationConfiguration;
        }
        public void setRuntimeInfo(RuntimeInfo runtimeInfo){
            this.runtimeInfo=runtimeInfo;
        }
        public RuntimeInfo getRuntimeInfo(){
            return this.runtimeInfo;
        }
    }
    private ConcurrentHashMap<Thread,ActionContext> contexts=new ConcurrentHashMap<>();
    public ActionContext getActionContext(){
        ActionContext actionContext=contexts.get(Thread.currentThread());
        if(actionContext==null){
            actionContext=new ActionContext();
            contexts.put(Thread.currentThread(), actionContext);
        }
        return actionContext;
    }
*/
}