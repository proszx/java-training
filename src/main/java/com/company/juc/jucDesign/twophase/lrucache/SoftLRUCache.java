package com.company.juc.jucdesign.twophase.lrucache;
import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class SoftLRUCache<K,V>{
    private final LinkedList<K> keyList=new LinkedList<>();
    private final Map<K,SoftReference<V>> cache=new HashMap<>();
    private final int cap;
    private final CacheLoader<K,V> cacheLoader;
    public SoftLRUCache(int cap,CacheLoader<K,V> cacheLoader){
        this.cap=cap;
        this.cacheLoader=cacheLoader;
    }
    public void put(K key,V value){
        if(keyList.size()>=cap){
            K eldesKey=keyList.removeFirst();
            cache.remove(eldesKey);
        }
        if(keyList.contains(key)){
            keyList.remove(key);
        }
        keyList.addLast(key);
        cache.put(key,new SoftReference<>(value));
    }
    public V get(K key){
        V value;
        boolean success=keyList.remove(key);
        if(!success){
            value=cacheLoader.load(key);
            this.put(key, value);
        }else{
            value=cache.get(key).get();
            keyList.addLast(key);
        }
        return value;
    }
    @Override
    public String toString(){
        return this.keyList.toString();
    }
    public static void main(String[] args) {
        LRUCache<Integer,Reference> cache=new LRUCache<>(100,key->new Reference());
       System.out.println(cache);
       IntStream.range(0,Integer.MAX_VALUE).forEach(i -> new Thread(()->{cache.get(i);
       try {
		TimeUnit.SECONDS.sleep(1);
	} catch (InterruptedException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
       System.out.println(i+"sorted at chache");}).start());

       for(int i=0;i<Integer.MAX_VALUE;i++){
           cache.get(i);
           try {
			TimeUnit.SECONDS.sleep(1);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
       System.out.println(i+"sorted at chache");
       }
    }


}