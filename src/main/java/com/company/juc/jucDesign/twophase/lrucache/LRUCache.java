package com.company.juc.jucdesign.twophase.lrucache;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

public class LRUCache<K,V> {
    private final LinkedList<K> keyList=new LinkedList<>();
    private final Map<K,V> cache=new HashMap<>();
    private final int cap;
    private final CacheLoader<K,V> cacheLoader;
    public LRUCache(int cap,CacheLoader<K,V> cacheLoader){
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
        cache.put(key, value);
    }
    public V get(K key){
        V value;
        boolean success=keyList.remove(key);
        if(!success){
            value=cacheLoader.load(key);
            this.put(key, value);
        }else{
            value=cache.get(key);
            keyList.addLast(key);
        }
        return value;
    }
    @Override
    public String toString(){
        return this.keyList.toString();
    }
    public static void main(String[] args) {
        LRUCache<String,Reference> cache=new LRUCache<>(5,key->new Reference());
        cache.get("Alex");
        cache.get("Alexb");
        cache.get("Alexc");
        cache.get("Alexd");
        cache.get("Alexe");

        cache.get("july");
        System.out.println(cache.toString());
    }

}