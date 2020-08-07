
package com.company.juc.jucdesign.applicationcontext;
public class ActionContext {
     /**private static ThreadLocal<Context> contexts=ThreadLocal.withInitial(Context::new);
     public static Context get(){
         return contexts.get();
     }
     static class Context{
         private Configuration configuration;
         private OtherResource otherResource;
         public Configuration getConfiguration(){
            return this.configuration;
         }
         public void setConfiguration(Configuration configuration){
            this.configuration=configuration;
         }
         public OtherResource getOtherResource(){
             return this.otherResource;
         }
         public void setOtherResource(OtherResource otherResource){
             this.otherResource=otherResource;
         }
     }
     // TODO 独立封装
     private static ThreadLocal<Configuration> conf=ThreadLocal.withInitial(Configuration::new);
     private static ThreadLocal<OtherResource> oRe=ThreadLocal.withInitial(OtherResource::new);
     public static void setConfiguration(Configuration conf){
        conf.set(conf);
     }
     public static Configuration getConfiguration(){
        return conf.get();
     }
     public static void setOtherResource(OtherResource ore){
        oRe.set(ore);
     }
     public static OtherResource getOtherResource(){
        return oRe.get();
     }*/

}
