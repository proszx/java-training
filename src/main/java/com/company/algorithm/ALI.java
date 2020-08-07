package com.company.algorithm;

import java.lang.reflect.Array;
import java.util.*;
public class ALI {
    public static class test1 {
        public static void main(String[] args) {
            // write your code here
            Scanner sc = new Scanner(System.in);
            int n = Integer.valueOf(sc.nextLine());
            int[] tmp = new int[n];
            for (int i = 0; i < n; i++) {
                tmp[i] = Integer.valueOf(sc.nextLine().trim());
            }
            while (n-- > 0) {
                int res = 2;
                for (int i = 0; i < 32; i++) {
                    if (((tmp[n] >> i) & 1) != 0) {
                        res *= 2;
                    }
                }
                if (tmp[n] == Integer.MAX_VALUE) {
                    System.out.println(res / 2);
                } else {
                    System.out.println(res);
                }
            }

        }
    }
    public  static  class test2{
        static  class Node{
            int k;
            int v;
            public Node(int k,int v){
                this.k=k;
                this.v=v;
            }
        }
        public static void main(String[] args) {
            // write your code here
            Scanner sc=new Scanner(System.in);
            String line=sc.nextLine();
            String[] line1=line.trim().split(" ");
            //没有馅料
            int n=Integer.valueOf(line1[0]);
            int m=Integer.valueOf(line1[1]);
            int c0=Integer.valueOf(line1[2]);
            int d0=Integer.valueOf(line1[3]);
            //有馅料
            int[][] tmp=new int[1][4];
            List<Node> nodeStream=new ArrayList<>();
            for(int i=0;i<n;i++){
                int a=0,b=0,c=0,d=0;

                a=sc.nextInt();
                b=sc.nextInt();
                c=sc.nextInt();
                d=sc.nextInt();

                int cnt=a/b,k=1;
                while(cnt>0){
                    if(cnt>=k){
                        nodeStream.add(new Node(k*c,k*d));
                        cnt-=k;
                        k*=2;
                    }else{
                        nodeStream.add(new Node(cnt*c,cnt*d));
                        cnt=0;
                    }
                }
            }
            sc.close();
            int[] dp=new int[1010];
            for(int i=0;i<nodeStream.size()+1;i++){
                if(i==nodeStream.size()){
                    for(int j=c0;j<=n;j++){
                        dp[j]=Math.max(dp[j],dp[j-c0]+d0);
                    }
                }else{
                    for(int j=n;j>=nodeStream.get(i).k;j--){
                        dp[j]=Math.max(dp[j],dp[j-nodeStream.get(i).k]+nodeStream.get(i).v);
                    }
                }
            }
            System.out.println(dp[n]);
        }
    }


}
