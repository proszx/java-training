package com.company.BloomFilter;

import java.util.BitSet;

public class MyBloomFilter {
    private static  final int DEFAULT_SIZE=2<<24;
    private static final int[] SEEDS=new int[]{2,13,26,15,17,33};
    private BitSet set=new BitSet(DEFAULT_SIZE);
    private SimpleHash[] func=new SimpleHash[SEEDS.length];
    public MyBloomFilter(){
        for(int i=0;i<SEEDS.length;i++){
            func[i]=new SimpleHash(DEFAULT_SIZE,SEEDS[i]);
        }
    }
    public  void add(Object value){
        for(SimpleHash simpleHash:func){
            set.set(simpleHash.hash(value),true);
        }
    }
    /**
     * 判断指定元素是否存在于位数组
     */
    public boolean contains(Object value) {
        boolean ret = true;
        for (SimpleHash f : func) {
            ret = ret && set.get(f.hash(value));
        }
        return ret;
    }
    public class SimpleHash{
        private int size;
        private int seed;
        SimpleHash(int size,int seed){
            this.size=size;
            this.seed=seed;
        }
        public int hash(Object value){
            int h;
            return (value == null) ? 0 : Math.abs(seed * (size - 1) & ((h = value.hashCode()) ^ (h >>> 16)));
        }

    }
}
