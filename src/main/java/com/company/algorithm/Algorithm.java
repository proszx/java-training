package com.company.algorithm;

import java.util.*;

public class Algorithm{

    private static class ListNode{
        int val;
        ListNode next;
        ListNode(int x){
            val = x;
        }
    }
    private static class TreeNode{
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x){
            val = x;
        }
        TreeNode(int val, TreeNode left, TreeNode right) {
             this.val = val;
             this.left = left;
             this.right = right;
        }
    }
    private static class TreeLinkNode{
        int val;
        TreeLinkNode next;
        TreeLinkNode right;
        TreeLinkNode left;
        TreeLinkNode(int x){
            val=x;
        }
        TreeLinkNode(int _val, TreeLinkNode _left, TreeLinkNode _right, TreeLinkNode _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }
    private static class TreeRandomNode{
        int val;
        TreeRandomNode random;
        TreeRandomNode right;
        TreeRandomNode left;
        TreeRandomNode(int x){
            val=x;
        }
    }
    private static class LinkRandomNode{
        int val;
        LinkRandomNode next;
        LinkRandomNode random;

        public LinkRandomNode(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
        public LinkRandomNode(int val,LinkRandomNode next,LinkRandomNode random) {
            this.val = val;
            this.next = next;
            this.random = random;
        }
    }
    public static void main(String[] args)
    {
       // String s="barfoothefoobarman";
       ListNode head=new ListNode(2);
       head.next=new ListNode(1);
       head.next.next=new ListNode(6);
       head.next.next.next=new ListNode(4);
    //    ListNode re=ListNodeOperations.rotateRight(head,3);
    //    System.out.println("leetcode 24");
    //    ListNode res=ListNodeOperations.removeNthFromEnd(head,3);
    //    System.out.println("leetcode 25");
    //    String ress=StringOperations.getPermutation(3,3);
    //    System.out.println("leetcode 60");

    //    String tmp2=StringOperations.minWindow("ADOBECODEBANC", "ABC");
    //     System.out.println("leetcode 76");

    //     int[] tmp3={1,2,3};
    //     List<List<Integer>> result=BackTrace.subsetsBack(tmp3);
    //     System.out.println("leetcode 78");
    //     char[][] board ={{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}};
    //     String word="ABCCED";
    //     boolean isExist=BackTrace.exist(board, word);
    //     System.out.println("leetcode 79");

    //     String s="25525525555";
    //     List<String> results=BackTrace.IPProblems.restoreIpAddresses(s);
    //     System.out.println(result);

    //     int[] arr={-10,-3,0,5,9};
    //     TreeNode root=BSTOperations.sortedArrayToBST(arr);
    //     System.out.println(result);

    //     int[] test={2,2,3,2};
    //     int resultsofDup= MathsOperations.singleNumberofThree(test);
    //     System.out.println(resultsofDup);

    //     String worsd="catsanddog";
    //     List<String> wordDict=new ArrayList<>();
    //     String[] c={"cat", "cats", "and", "sand", "dog"};
    //     wordDict.addAll(Arrays.asList(c));
    //     List<String> wordSplit=StringOperations.wordBreakList(worsd, wordDict);
    //     System.out.println(wordSplit);

       // ListNodeOperations.reorderList(head);
       // System.out.println(head);
       //ListNode result=ListNodeOperations.insertionSortList(head);
       //System.out.println(result);
       int[] nums={2,1};
    //    int result=ArraysOperations.minSubArrayLen(4,nums);
    //     System.out.println(result);
    //    boolean result2=ArraysOperations.containsNearbyAlmostDuplicate(nums,1,1);
    //    System.out.println(result2);
        int area=MathsOperations.computeArea(-2,-2,2,2,1,-3,3,-1);
        System.out.println(area);
        int numCourses=4;
        int[][] prerequisites={{1,0},{2,0},{3,1},{3,2}};
        int[] course=GraphOperations.findOrder(numCourses, prerequisites);
        int[] rating={2,5,3,4,1};
        //System.out.println(numTeams(rating));
    }
    static class BitCal{
        /**
         * 不用加减乘除完成加法
         * @param a
         * @param b
         * @return
         */
        public static int getSum(int a, int b) {
            return b==0?a:getSum(a^b,a&b<<1);
        }

        /**
         * 统计从 0 ~ n 每个数的二进制表示中 1 的个数
         * @param num
         * @return
         */
        public int[] countBit(int num){
            int[] bit=new int[num+1];
            for(int i=1;i<=num;i++){
                bit[i]=bit[i&(i-1)]+1;
            }
            return bit;
        }
        /**
         * 求一个数的补码
         */
        public int findComplement(int num){
            if(num==0) return 1;
            int mask=1<<30;
            while((num^mask)==0){
                mask>>=1;
            }
            mask=(mask<<1)-1;
            return num^mask;
        }
        public int findComplementSimple(int num){
            int mask=num;
            mask|=mask>>1;
            mask|=mask>>2;
            mask|=mask>>4;
            mask|=mask>>8;
            mask|=mask>>16;
            return num^mask;
        }

        /**
         *判断一个数的位级表示是否不会出现连续的 0 和 1
         * @param n
         * @return
         */
        public boolean hasAlternatingBits(int n){
            int a=(n^(n>>1));
            return (a&(a+1))==0;
        }

    }
    public static class GraphOperations {
        public static int[] findOrder(int numCourses, int[][] prerequisites) {
            List<Integer>[] lists=new ArrayList[numCourses];
            //保存图得入度
            int[] pt=new int[numCourses];
            //构造图
            for(int[] p:prerequisites) {
                pt[p[0]]++;
                if(lists[p[1]]==null) {
                    lists[p[1]]=new ArrayList<>();
                }
                lists[p[1]].add(p[0]);
            }
            //创建队列保存数据
            Queue<Integer> queue=new LinkedList<>();
            //先存入没有入度得节点
            for(int i=0;i<numCourses;i++){
                if(pt[i]==0){
                    queue.add(i);
                }
            }
            //
            int[] res=new int[numCourses];

            int idx=0;
            //当节点不为空
            while(!queue.isEmpty()){
                //计算图得大小
                int size=queue.size();
                while(size-->0){
                    //得到入度最少的节点
                    int p=queue.poll();
                    //保存为结果
                    res[idx++]=p;
                    //得到第一个列表
                    List<Integer> list=lists[p];
                    if(list==null){
                        continue;
                    }
                    //
                    for(int val:list){
                        //得到节点 减少入度
                        pt[val]--;
                        if(pt[val]==0){
                            //遍历减少 当入度为0 加入队列 //本质上是bfs
                            queue.add(val);
                        }
                    }
                }
            }
            //可能会有多个正确的顺序，你只要返回一种就可以了
            //。如果不可能完成所有课程，返回一个空数组。
            return idx==numCourses?res:new int[0];
        }
    }
    public static class ListNodeOperations {
        /**
         * 两个链表的第一个公共节点
         * @param pHead1
         * @param pHead2
         * @return
         */
        public ListNode findFirstCommonNode(ListNode pHead1, ListNode pHead2) {
            ListNode p1=pHead1,p2=pHead2;
            while(p1!=p2){
                p1=p1==null?pHead2:p1.next;
                p2=p2==null?pHead1:p2.next;
            }
            return p1;
        }
            /**
             * leetcode 876
             * 给定一个带有头结点 head 的非空单链表，返回链表的中间结点。
             * 如果有两个中间结点，则返回第二个中间结点
             * @param head
             */
        public ListNode middleNode(ListNode head) {
            ListNode fast=head,slow=head;
            while(fast!=null&&fast.next!=null){
                fast=fast.next.next;
                slow=slow.next;
            }
            return slow;
        }
        /**
         * 请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点，你将只被给定要求被删除的节点。
         * @param
         */
        public void deleteNode(ListNode node) {
            node.val=node.next.val;
            node.next=node.next.next;
        }
        /**
         * 链表回文 首先节点要遍历 前遍历数s1=s1*10+val  后遍历数 s2=s2+val*t t*=10 位数变换.
         * 或者linkedlist addfirst addlast 最后tostring进行比较
         * @param head
         */
        public boolean isListNodePalindrome(ListNode head) {
            float s1=0,s2=0;
            float t=1;
            if(head==null) return true;
            while(head!=null){
                s1=s1*10+head.val;
                s2=s2+head.val*t;
                head=head.next;
                t*=10;
            }
            return s1==s2;
        }
        /**
         * 删除链表中等于给定值 val 的所有节点。
         * @param head,n
         */
        public ListNode removeElements(ListNode head, int val) {
            if(head==null){
                return null;
            }
            head.next=removeElements(head.next,val);
            return head.val==val?head.next:head;
        }
        public ListNode removeElementsA(ListNode head, int val) {
            if(head==null){
                return null;
            }
            ListNode dummy=new ListNode(-1);
            dummy.next=head;
            ListNode cur=dummy;
            while(cur.next!=null){
                if(cur.next.val==val){
                    cur.next=cur.next.next;
                }else{
                    cur=cur.next;
                }

            }
            return dummy.next;
        }
        /**
         * 编写一个程序，找到两个单链表相交的起始节点。
         * @param headA
         * @param headB
         * @return
         */
        public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            if(headA == null || headB == null){
                return null;
            }
            ListNode pA=headA,pB=headB;
            while(pA!=pB){
                pA=pA==null?headB:pA.next;
                pB=pB==null?headA:pB.next;
            }
            return pA;
        }
        /**
         * 归并排序
         * @param head
         */
        public static ListNode sortList(ListNode head) {
            return head==null?null:mergeSort(head);
        }
        private static ListNode mergeSort(ListNode head) {
            if(head.next==null){
                return head;
            }
            ListNode p=head,q=head,pre=null;
            while(q!=null&&q.next!=null){
                pre=p;
                p=p.next;
                q=q.next.next;
            }
            pre.next=null;
            ListNode l=mergeSort(head);
            ListNode r=mergeSort(p);
            return merge(l,r);
        }
        private static ListNode merge(ListNode l,ListNode r) {
            ListNode dummy = new ListNode(0),cur=dummy;
            while(l!=null && r!=null){
                if(l.val<=r.val){
                    cur.next=l;
                    cur=cur.next;
                    l=l.next;
                }else{
                    cur.next=r;
                    cur=cur.next;
                    r=r.next;
                }
                if(l!=null){
                    cur.next=l;
                }
                if(r!=null){
                    cur.next=r;
                }
            }
            return dummy.next;
        }
        /**
         * 插入排序是迭代的，每次只移动一个元素，
         * 直到所有元素可以形成一个有序的输出列表
         * 每次迭代中，插入排序只从输入数据中移除
         * 一个待排序的元素，找到它在序列中适当的
         * 位置，并将其插入重复直到所有输入数据插
         * 入完为止。
         * @param head
         * @return
         * 也可以用链表做
         * 选择一个大于当前值 就加入
         */
        public static ListNode insertionSortList(ListNode head) {
            ListNode dummy = new ListNode(0),prev = head;
            dummy.next=head;
            while(head!=null&&head.next!=null){
                if(head.val<=head.next.val){
                    head=head.next;
                    continue;
                }
                prev=dummy;

                while(prev.next.val<head.next.val){
                    prev=prev.next;
                }
                ListNode cur=head.next;
                head.next=cur.next;
                cur.next=prev.next;
                prev.next=cur;
            }
            return dummy.next;
        }
        /**
         * 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。 k 是一个正整数，它的值小于或等于链表的长度。
         * 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
         *
         * @param head
         * @param k
         * @return
         */
        public static ListNode reverseKGroup(ListNode head, int k) {
            // return head;
            ListNode dummy = new ListNode(0), prev = dummy, cur = head, next;
            dummy.next = head;
            int len = 0;
            while (head != null) {
                len++;
                head = head.next;
            }
            head = dummy.next;
            for (int i = 0; i < len; i++) {
                for (int j = 0; j < k - 1; j++) {
                    next = cur.next;
                    cur.next = next.next;
                    next.next = prev.next;
                    prev.next = next;
                }
                prev = cur;
                cur = prev.next;
            }
            return dummy.next;
        }
        /**
         * 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
         * 一趟扫描
         * 所以不方便使用
         *
         * @param head,m,n
         */
        public ListNode reverseBetween(ListNode head, int m, int n) {
            ListNode dummyH = new ListNode(-1);
            dummyH.next = head;
            ListNode sper = dummyH;
            for (int i = 1; i < m; i++) {
                sper = sper.next;
            }
            ListNode prev = null;
            ListNode cur = sper.next;
            for (int i = 0; i <= n - m; i++) {
                ListNode next = cur.next;
                cur.next = prev;
                prev = cur;
                cur = next;
            }
            sper.next.next = cur;
            sper.next = prev;
            return dummyH.next;
        }
        /**
         * 给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
         * 将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…
         */
        public static void reorderList(ListNode head) {
            LinkedList<ListNode> queue = new LinkedList<>();
            ListNode cur=head;
            while(cur!=null) {
                queue.addLast(cur);
                cur=cur.next;
            }
            while(!queue.isEmpty()){
                if(cur==null){
                    cur=queue.pollFirst();
                }else{
                    cur.next=queue.pollFirst();
                    cur=cur.next;
                }
                cur.next=queue.pollLast();
                cur=cur.next;
            }
            if(cur!=null){
                cur.next=null;
            }
        }
        /**
         * 判断链表是否是环形链表
         */
        public static boolean hasCycle(ListNode head){
            ListNode p = head, p2 = head;
            boolean hasCycle = false;
            while (p2.next != null && p2.next.next != null) {
                p = p.next;
                p2 = p2.next.next;
                if (p == p2) {
                    hasCycle = true;
                    break;
                }
            }

            // 步骤二：若有环，找到入环开始的节点
            if (hasCycle) {
                ListNode q = head;
                while (p != q) {
                    p = p.next;
                    q = q.next;
                }
                return true;
            } else
                return false;
        }
        /**
         * 找到环形链表的开始节点
         */
        public ListNode detectCycle(ListNode head) {
            if(head==null||head.next==null) return null;
            ListNode fast=head.next;
            ListNode slow=head;
            while(fast!=slow){
                if(fast.next==null||fast.next.next==null) return null;
                fast=fast.next.next;
                slow=slow.next;
            }
            return slow.next;
        }


        /**
         * 给定一个链表和一个特定值 x，对链表进行分隔， 使得所有小于 x 的节点都在大于或等于 x 的节点之前。
         *
         * @param head，x
         * @return Dummy Node 的目的是为了减少额外的 null 检查 dummy node一般指向链表的头结点head,
         *         使得链表的每一个节点都有一个前驱, 方便操作,即使链表的头部发生变化, 我们不需要修改head,只需要找到dummy.next
         */
        public static ListNode partition(ListNode head, int x) {
            if (head == null) {
                return null;
            }
            ListNode dummyLeft = new ListNode(-1);
            ListNode dummyRight = new ListNode(-1);
            ListNode curLeft = dummyLeft;
            ListNode curRight = dummyRight;
            while (head != null) {
                if (head.val < x) {
                    curLeft.next = new ListNode(head.val);
                    curLeft = curLeft.next;
                } else {
                    curRight.next = new ListNode(head.val);
                    curRight = curRight.next;
                }
                head=head.next;
            }
            curLeft.next = dummyRight.next;
            return dummyRight.next;
        }

        /**
         * 给定一个排序链表，删除所有重复的元素， 使得每个元素只出现一次。
         *
         * @param head
         */
        public static ListNode deleteDuplicatesI(ListNode head) {
            if (head == null || head.next == null) {
                return head;
            }
            head.next = deleteDuplicatesI(head.next);
            if (head.val == head.next.val) {
                head = head.next;
            }
            return head;
        }

        /**
         * 给定一个排序链表，删除所有含有重复数字的节点， 只保留原始链表中
         *  没有重复出现 的数字
         *
         * @param head
         */
        public static ListNode deleteDuplicatesII(ListNode head) {
            if (head == null || head.next == null) {
                return head;
            }
            ListNode next = head.next;
            if (head.val == next.val) {
                while (next != null && head.val == next.val) {
                    next = next.next;
                }
                head = deleteDuplicatesII(next);
            } else {
                head.next = deleteDuplicatesII(next);
            }
            return head;
        }

        /**
         * 将两个升序链表合并为一个新的 升序 链表并返回。 新链表是通过拼接给定的两个链表的所有节点组成的。
         *
         * @param l1
         * @param l2
         * @return
         */
        public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
            if (l1 == null)
                return l2;
            if (l2 == null)
                return l1;

            ListNode head = null;
            if (l1.val <= l2.val) {
                head = l1;
                head.next = mergeTwoLists(l1.next, l2);
            } else {
                head = l2;
                head.next = mergeTwoLists(l1, l2.next);
            }
            return head;
        }

        /**
         * 将多个升序链表合并为一个新的 升序 链表并返回。 新链表是通过拼接给定的两个链表的所有节点组成的。
         *
         * @param lists
         * @return
         * 2020年7月31日 532
         */
        public static ListNode mergeKLists(ListNode[] lists) {
            if (lists.length == 0)
                return null;
            if (lists.length == 1)
                return lists[0];
            if (lists.length == 2) {
                return mergeTwoLists(lists[0], lists[1]);
            }

            int mid = lists.length / 2;
            ListNode[] l1 = new ListNode[mid];
            for (int i = 0; i < mid; i++) {
                l1[i] = lists[i];
            }

            ListNode[] l2 = new ListNode[lists.length - mid];
            for (int i = mid, j = 0; i < lists.length; i++, j++) {
                l2[j] = lists[i];
            }

            return mergeTwoLists(mergeKLists(l1), mergeKLists(l2));
        }

        /**
         * 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
         *
         * @param head
         * @return
         */
        public static ListNode swapPairs(ListNode head) {
            if (head == null || head.next == null) {
                return head;
            }
            ListNode tmp = head.next;
            head.next = swapPairs(tmp.next);
            tmp.next = head;
            return tmp;
        }



        public static ListNode removeNthFromEnd(ListNode head, int n) {
            if (head == null || head.next == null)
                return null;
            ListNode fast = head, slow = head;
            for (int i = 0; i < n; i++) {
                fast = fast.next;
            }
            if (fast == null) {
                return head.next;
            }
            while (fast.next != null) {
                fast = fast.next;
                slow = slow.next;
            }
            slow.next = slow.next.next;
            return head;
        }

        /**
         * 给定一个链表，旋转链表， 将链表每个节点向右移动 k 个位置，其中 k 是非负数。
         *
         * @param head
         * @param k
         * @return
         */
        public static ListNode rotateRight(ListNode head, int k) {
            // return head;
            if (head == null || k == 0) {
                return head;
            }
            ListNode cur = head;
            ListNode tail = null;
            int length = 1;
            while (cur.next != null) {
                cur = cur.next;
                length++;
            }
            int l = length - (k % length);
            tail = cur;
            cur.next = head;
            cur = head;
            for (int i = 0; i < l; i++) {
                cur = cur.next;
                tail = tail.next;
            }
            tail.next = null;
            return cur;
        }
    }

    public static class ArraysOperations {
        /**
         * 数字在排序数组中出现的次数
         */
        public int GetNumberOfK(int[] nums, int K) {
           int left= Arrays.binarySearch(nums,K);
           int right= Arrays.binarySearch(nums,K+1);
           return (left==nums.length||nums[left]!=K)?0:right-left;
        }
        private int binarySearch(int[] nums,int k){
            int l=0,r=nums.length-1;
            while(l<r){
                int mid=l+(r-l)/2;
                if(nums[mid]==k){
                    return mid;
                }else if(nums[mid]>k){
                    r=mid;
                }else{
                    l=mid+1;
                }
            }
            return l;
        }
        /**
         * 逆序对
         */
        private int cnt=0;
        private int[] tmp;
        public int inversePairs(int[] nums) {
            tmp=new int[nums.length];
            mergeSort(nums,0,nums.length-1);
            return cnt%100000007;
        }
        private void mergeSort(int[] nums,int l,int r){
            if(l>r){
                return;
            }
            int mid=l+(r-l)/2;
            mergeSort(nums,l,mid);
            mergeSort(nums,mid+1,r);
            merge(nums,l,mid,r);
        }
        private void merge(int[] nums,int left,int mid,int right){
            int i=left,j=mid+1,k=l;
            while(i<=mid||j<=right){
                if(i>mid){
                    tmp[k++]=nums[j++];
                }
                else if(j>right){
                    tmp[k++]=nums[i++];
                }
                else if(nums[i]<nums[j]){
                    tmp[k++]=nums[i++];
                }else{
                    tmp[k++]=nums[j++];
                    cnt+=mid-i+1;
                }
            }
            for(k=left;i<k;i++){
                nums[i]=nums[k];
            }
        }

        /**
         * 给定高度m 、宽度n 的一张 m * n的乘法表，以及正整数k，你需要返回表中第k 小的数字。
         * 1. 用堆实现
         * @param m
         * @param n
         * @param k
         * @return
         * 会出现超时的情况，最好的方法是二分搜索
         */
        public int findKthNumberByHeap(int m, int n, int k) {
            PriorityQueue<Integer> queue=new PriorityQueue<>((o1,o2)->(o2-o1));
            for(int i=1;i<=m;i++){
                for(int j=1;j<=n;j++){
                    if(queue.size()==k){
                        if(i*j<queue.peek()){
                            queue.poll();
                            queue.offer(i*j);
                        }
                    }else{queue.offer(i*j);}
                }
            }
            return queue.peek();
        }
        public int findKthNumberByBina(int m, int n, int k) {
            int i=1,j=m*n+1;
            while(i<j){
                int mid=i+(j-i)/2;
                int cur=getTheCount(m,n,mid);
                if(cur<k){
                    i=mid+1;
                }else {
                    j=mid-1;
                }
            }
            return i;
        }
        private int getTheCount(int m,int n,int mid){
            int cnt=0;
            for(int i=1;i<=m;i++){
                cnt+=Math.min(mid/i,n);
            }
            return cnt;
        }

        /**
         * 给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间
         * 包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，
         * 找出这个重复的数。
         */
        public int findDuplicate(int[] nums) {
            int fast=0,slow=0;
            while(true){
                fast=nums[nums[fast]];
                slow=nums[slow];
                if(slow==fast){
                    fast=0;

                while(nums[slow]!=nums[fast]){
                    fast=nums[fast];
                    slow=nums[slow];
                }
                return nums[slow];
            }
        }

    }
        /**
         * 把0移到最后
         */
        public void moveZeroes(int[] nums) {
            if(nums.length<=1||nums==null){
                return;
            }
            int index = 0;
            for(int i=0;i<nums.length; i++){
                if(nums[i]!=0){
                    nums[index++] =nums[i];
                }
            }
            for(int i=index; i<nums.length; i++){
                nums[i]=0;
            }
        }
        /**
         * 给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。
         */
        public int missingNumber(int[] nums) {
            int res=nums.length;
            for(int i=0;i<nums.length;i++){
                res^=nums[i];
                res^=i;
            }
            return res;
        }

        /**
         * 给定一个非空整数数组，除了某个元素只出现一次以外，
         * 其余每个元素均出现两次。找出那个只出现了一次的元素
         */
        public static int singleNumber(int[] nums) {
            int re=0;
            for(int num:nums){
                re^=num;
            }
            return re;

        }
        /**
         * 给定一个整数数组 nums，其中恰好有两个元素只出现一次，
         * 其余所有元素均出现两次。 找出只出现一次的那两个元素。
         * @param nums
         */
        public int[] singleNumberofTwo(int[] nums) {
            int num1=0,num2=0;
            int xor=0;
            for(int num : nums){
                xor^=num;
            }
            int bit=xor&(~xor+1);
            for(int num : nums){
                if((num&bit)==0){
                    num1^=num;
                }else{
                    num2^=num;
                }
            }

            return new int[]{num1,num2};
        }
        /**
         * 给定一个非空整数数组，除了某个元素只出现一次以外，
         * 其余每个元素均出现三次。找出那个只出现了一次的元素
         */
        public static int singleNumberofThree(int[] nums) {
            int a=0,b=0;
            for(int num:nums){
                b=(b^num)&~a;
                a=(a^num)&~b;
            }
            return b;
        }
        /**
         *
         * @param matrix
         * @param target
         * @return
         */
        public boolean searchMatrixs(int[][] matrix, int target) {
            if(matrix==null||matrix.length==0||matrix[0].length==0){
                return false;
            }
                int cols=matrix[0].length,rows=matrix.length-1;
                int c=cols-1,r=0;
                while(c>=0&&r<=rows){
                    if(matrix[r][c]==target){
                        return true;
                    }else if(matrix[r][c]>target){
                        c--;
                    }else{
                        r++;
                    }
                }
                return false;
        }
        public static class SlidingWindow{
            /**
             * 输出所有和为 S 的连续正数序列的个数
             * 滑动窗口
             * 和为sum的连续正整数序列
             *
             * 滑动窗口
             */
            public  int findSerialSum(int sum){
                List<List<Integer>> result=new ArrayList<>();
                int pre=1,pro=2;
                int cur=3;
                while(pro<sum){
                    if(cur>sum){
                        cur-=pre;
                        pre++;
                    }else if(cur<sum){
                        pro++;
                        cur+=pro;
                    }else {
                        List<Integer> list=new ArrayList<>();
                        for(int i=pre;i<pro;i++){
                            list.add(i);
                        }
                        result.add(new ArrayList<>(list));
                        cur-=pre;
                        pre++;
                        pro++;
                        cur+=pro;
                    }
                }
                return result.size();

            }
            /**
             * 滑动窗口的最大值
             * @param nums
             * @param k
             * @return
             */
            public int[] maxSlidingWindow(int[] nums, int k) {
                if (nums == null || nums.length<2){
                    return nums;
                }
                LinkedList<Integer> list=new LinkedList<>();
                //k个最大值
                int[] result = new int[nums.length-k+1];
                for(int i=0; i<nums.length;i++){
                    while(!list.isEmpty()&&nums[list.peekLast()]<=nums[i]){
                        list.pollLast();
                    }
                    list.addLast(i);
                    if(list.peek()<=i-k){
                        list.poll();
                    }
                    if(i-k+1>=0){
                        result[i-k+1]=nums[list.peek()];
                    }
                }
                return result;
            }
            /**
             *滑动窗口的大值
             */
            public ArrayList<Integer> maxInWindow(int[] nums,int k){
                ArrayList<Integer> re=new ArrayList<>();
                if(k>nums.length||k<1){
                    return re;
                }
                PriorityQueue<Integer> queue=new PriorityQueue<>((o1, o2) -> (o2-o1));
                for(int i=0;i<nums.length;i++){
                    queue.add(nums[i]);
                }
                re.add(queue.peek());
                for(int i=0,j=i+k;j<nums.length;i++,j++){
                    queue.remove(nums[i]);
                    queue.add(nums[j]);
                    re.add(queue.peek());
                }
                return re;
            }
        }


        /**
         * 给你一个长度为 n 的整数数组 nums，其中 n > 1，
         * 返回输出数组 output ，
         * 其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
         * @param nums
         * @return
         */
        public int[] productExceptSelf(int[] nums) {
            if(nums==null||nums.length==0){
                return new int[0];
            }
            int len=nums.length;
            int[] result=new int[len];
            int left=1;
            for(int i=0;i<len;i++){
                if(i>0){
                    left=left*nums[i-1];
                }
                result[i]=left;
            }
            int right=1;
            for(int i=len-1;i>=0;i--){
                if(i<len-1){
                    right=right*nums[i+1];
                }
                result[i]*=right;
            }
            return result;
        }
        /**
         * 找出最多的
         * @param nums
         * @return
         */
        public int majorityElement(int[] nums) {
            int cnt=0,last=0;
            for(int num:nums) {
                if(cnt==0){
                    last=num;
                }
                cnt=cnt+(last==num?1:-1);
            }
            return last;
        }
        /**
         * 给定一个大小为 n 的数组，找出其中所有出现超过 ⌊ n/3 ⌋ 次的元素。
         * 摩尔投票法
         * @param nums
         * @return
         */
        public List<Integer> majorityElementOfThree(int[] nums) {
            int cnt1=0,cnt2=0;
            int x=0,y=0;
            for(int i=0;i<nums.length; i++){
                if(cnt1==0){
                    x=nums[i];
                    cnt1++;
                }else if(cnt2==0&&nums[i]!=x){
                    y=nums[i];
                    cnt2++;
                }else{
                    if(nums[i]==x){
                        cnt1++;
                    }else if(nums[i]==y){
                        cnt2++;
                    }else{
                        cnt1--;
                        cnt2--;
                        if(cnt1==0){
                            x=y;
                            cnt1=cnt2;
                            cnt2=0;
                        }
                    }
                }
            }
                //上述过程是为了找出最多的两个数
                cnt1=0;
                cnt2=0;
                Set<Integer> list=new HashSet<Integer>();
                for(int i=0;i<nums.length;i++){
                    if(nums[i]==x){
                        cnt1++;
                    }
                    if(nums[i]==y){
                        cnt2++;
                    }
                }
            if(cnt1>nums.length/3){
                list.add(x);
            }
            if(cnt2>nums.length/3){
                list.add(y);
            }
            return new ArrayList<>(list);
        }
        /**
         * 给定一个无重复元素的有序整数数组，返回数组区间范围的汇总。
         * @param nums
         * @return
         *
         */
        public List<String> summaryRanges(int[] nums) {
            List<String> result = new ArrayList<String>();
            for(int i = 0; i < nums.length; i++){
                int idx= i;
                while(idx+1 < nums.length&&nums[idx+1]==nums[idx]+1){
                    idx++;
                }
                if(idx==i){
                    result.add(nums[idx]+"");
                }else{
                    result.add(nums[i]+"->"+nums[idx]);
                    i=idx;
                }
            }
            return result;
        }

        //这里使用hashmap部分用例会不通过，原因是因为[2,1] 1 1
        //这样的测试用例 达到上边界就自动退出
        //因此此处采用TreeSet加上滑动窗口得方法
        //，这类题目 滑动窗口才是通解
        //ceiling方法用来返回在此设定为大于或等于给定的元素的最小元素，或null，如果不存在这样的元素。
        public static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
            TreeSet<Long> set = new TreeSet<>();
            for(int i=0; i<nums.length; i++){
                Long ceiling=set.ceiling((long)nums[i]-(long)t);
                if(ceiling!=null&&ceiling<=(long)nums[i]+(long)t){
                    return true;
                }
                set.add((long)nums[i]);
                if(set.size()==k+1){
                    set.remove((long)nums[i-k]);
                }
            }
            return false;
      }
        /**
         * 给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 
         * i 和 j，使得 nums [i] = nums [j]，
         * 并且 i 和 j 的差的 绝对值 至多为 k。
         * @param nums,k
         */
        public boolean containsNearbyDuplicate(int[] nums, int k) {
            if(k==0||nums.length==0) return false;
            int n=nums.length;
            HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
            for(int i=0;i<n;i++){
                if(map.get(nums[i])!=null&&(i-map.get(nums[i])<=k)){
                    return true;
                }
                map.put(nums[i],i);
            }
            return false;
        }

        private static class Point{
            int pos;
            int height;
            Point(int pos, int height){
                this.pos=pos;
                this.height=height;
            }
        }
        /**
         * 城市天际线
         * @param buildings
         * @return
         */
        public List<List<Integer>> getSkyline(int[][] buildings) {
            List<List<Integer>> resList = new ArrayList<>();
            if (buildings == null || buildings.length == 0 || buildings[0] == null || buildings[0].length == 0) {
                return resList;
            }
            //定义比较器，先按pos从小到大排序，pos相等，按height从小到大排
            PriorityQueue<Point> queue = new PriorityQueue<>((o1, o2) -> {
                if (o1.pos != o2.pos) {
                    return o1.pos - o2.pos;
                }
                if (o1.height != o2.height) {
                    return o1.height - o2.height;
                }
                return 0;
            });
            //生成queue，<第一个位置，负高度>，<第二个位置，正高度>
            for (int i = 0; i < buildings.length; i++) {
                queue.offer(new Point(buildings[i][0], -buildings[i][2]));
                queue.offer(new Point(buildings[i][1], buildings[i][2]));
            }
            PriorityQueue<Integer> maxHeightQueue = new PriorityQueue<>(Comparator.reverseOrder());
            maxHeightQueue.offer(0);
            int prePeak = maxHeightQueue.peek();
            while (!queue.isEmpty()) {
                //当当前高度是负数，说明是上升的，加到maxHeightQueue，反之移除掉
                Point point = queue.poll();
                if (point.height < 0) {
                    maxHeightQueue.offer(-point.height);
                } else {
                    maxHeightQueue.remove(point.height);
                }
                int curPeak = maxHeightQueue.peek();
                if (curPeak != prePeak) {
                    resList.add(Arrays.asList(point.pos, curPeak));
                    prePeak = curPeak;
                }
            }
            return resList;
        }
        /**
         * 给定一个整数数组，判断是否存在重复元素。
         * 如果任意一值在数组中出现至少两次，
         * 函数返回 true 。如果数组中每个元素都不相同，
         * 则返回 false 。
         * @param nums
         * @return
         */
        public boolean containsDuplicate(int[] nums) {
            return Arrays.stream(nums).distinct().count()!=nums.length;
        }
        /**
         * 在未排序的数组中找到第 k 个最大的元素。
         * 请注意，你需要找的是数组排序后的
         * 第 k 个最大的元素，而不是第 k 个不同的元素。
         * @param nums
         * @param k
         * @return
         */
        public int findKthLargest(int[] nums, int k) {
            Arrays.sort(nums);
            return nums[nums.length-k];
        }
        //三路快排实现
        public int findKthLargestQS(int[] nums, int k){
            quickSort(nums,0,nums.length-1,k);
            return nums[k-1];
        }
        private void quickSort(int[] nums, int l, int r,int k){
            if(l>=r) return;
            int pivot=nums[l];//选择标定点
            int lt=l;
            int gt=r+1;
            int i=l+1;
            while(i<gt){
                if(nums[i]>pivot){
                    int tmp=nums[i];
                    nums[i]=nums[lt+1];
                    nums[lt+1]=tmp;
                    ++lt;
                    ++i;
                }else if(nums[i]<pivot){
                    int tmp=nums[i];
                    nums[i]=nums[gt-1];
                    nums[gt-1]=tmp;
                    --gt;
                }else{
                    ++i;
                }
            }
            int tmp=nums[lt];
            nums[lt]=nums[l];
            nums[l]=tmp;
            if(lt>k-1)
                quickSort(nums, l, lt-1, k);
            else
                quickSort(nums,gt,r,k);
        }
        /**
         * 给定一个含有 n 个正整数的数组和一个正整数 s ，
         * 找出该数组中满足其和 ≥ s 的长度最小的连续子数组
         * ，并返回其长度。如果不存在符合条件的连续子数组，返回 0。
         * @param s
         * @param nums
         * @return
         * 滑动窗口
         */
        public static int minSubArrayLen(int s, int[] nums) {
            int i=0;
            int sum=0;
            int len=0;
            for(int j=0; j<nums.length; j++){
                sum+=nums[j];
                while(sum>=s){
                    len=len==0?j-i+1:Math.min(len,j-i+1);
                    sum-=nums[i++];
                }
            }
            return len;
        }
        /**
         * 给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。
         * @param nums，k
         */
        public void rotate(int[] nums, int k) {
            int n=nums.length;
            k%=n;
            reverse(nums,0,n-1);//合并排序的思想
            reverse(nums,0,k-1);
            reverse(nums,k,n-1);
        }
        private void reverse(int[] nums,int i,int j){
            while(i<j){
                int temp=nums[i];
                nums[i++]=nums[j];
                nums[j--]=temp;
            }
        }
        /**
         *
         * @param n
         * @return
         */
        public int trailingZeroes(int n) {
            if(n<5){
                return 0;
            }else{
                return n/5+trailingZeroes(n/5);
            }
        }


        /**
         * 给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
         * 升序数列 二分操作 两个数 双职能
         * @param numbers
         * @param target
         * @return
         */
        public int[] twoSum(int[] numbers, int target) {
            int l=0,r=numbers.length-1,ans;
            while(l<r){
                ans=numbers[l]+numbers[r];
                if(ans==target){
                    return new int[]{l+1,r+1};
                }else if(ans>target){
                    r--;
                }else{
                    l++;
                }
            }
            return new int[]{};
        }
        /**
         * 寻找最大间隔；
         * @params nums
         */
        public static int maximumGap(int[] nums) {
            if(nums.length<2) return 0;
            int max=0;
            Arrays.sort(nums);
            for(int i=0;i<nums.length-1;i++){
                max=Math.max(max,nums[i+1]-nums[i]);
            }
            return max;
        }
        /**
         * 寻找峰值
         * @param nums
         */
        public int findPeakElement(int[] nums) {
            int l=0,r=nums.length-1;
            while(l<r){
                int mid=l+(r-l)/2;
                if(nums[mid]>nums[mid+1]){
                    r=mid;
                }else{
                    l=mid+1;
                }
            }
            return l;

        }
        /**
         * 旋转数组请找出其中最小的元素
         * @param nums
         * @return
         */
        public static int findMinDup(int[] nums) {
            int l=0, r = nums.length - 1;
            while(l<r){
                int mid = l + (r - l) / 2;
                if(nums[l]==nums[mid] && nums[mid]==nums[r]){
                    return minNumber(nums,l,r);
                }
                if(nums[mid]<=nums[r]){
                    r=mid;
                }else{
                    l=mid + 1;
                }
            }
            return nums[l];
        }
        private static int minNumber(int[] nums,int l,int r){
            for(int i=0; i<r; i++){
                if(nums[i]>nums[i+1]){
                    return nums[i+1];
                }
            }
            return nums[l];
        }
        /**
         * 旋转数组请找出其中最小的元素。
         *@param nums
         */
        public static int findMin(int[] nums) {
            int l=0,r=nums.length - 1;
            while(l<r){
                int mid=l+ (r - l) / 2;
                if(nums[mid] < nums[r]){
                    r=mid;
                }else{
                    l=mid-1;
                }
            }
            return nums[l];
        }
        static class DPLeftRight{
            /**
             * 每个孩子至少分配到 1 个糖果. 相邻的孩子中，评分高的孩子必须获得更多的糖果。
             *
             * @param ratings
             * @return
             */
            public static int candy(int[] ratings) {
                int n = ratings.length;
                int[] arr = new int[n];
                Arrays.fill(arr,1); // 先每人分一个
                for(int i = 1; i < n; i++) { // 从左到右分当前分数大于前一个就变为前一个加一
                    if(ratings[i] > ratings[i-1]) arr[i] = arr[i-1] + 1;
                }
                int res = arr[n-1];
                for(int i = n - 2; i >= 0; i--) {// 从右到左分，当前分数大于后一个，且当前拿到的糖比后一个少就变为后一个加一
                    if(ratings[i] > ratings[i+1] && arr[i] <= arr[i+1]) arr[i] = arr[i+1] + 1;
                    res += arr[i];
                }
                return res;
            }
            /**
             *  n 名士兵站成一排。每个士兵都有一个 独一无二 的评分 rating
             * 寻找递增递减序列
             * 升序降序分开考虑 不能用三指针解决 停止条件找不到
             * @param rating
             * @return
             */
            public int numTeams(int[] rating) {
                int n = rating.length;
                int res = 0;
                int[] dp1 = new int[n];
                int[] dp2 = new int[n];
                Arrays.fill(dp1, 1);
                Arrays.fill(dp2, 1);
                for(int i = 1; i < n; i++) {
                    int a = 1;
                    int b = 1;
                    for(int j = 0; j < i; j++) {
                        if (rating[i] > rating[j]) {
                            a++;
                            res += (dp1[j] - 1);
                        } else {
                            b++;
                            res += (dp2[j] - 1);
                        }
                    }
                    dp1[i] = a;
                    dp2[i] = b;
                }
                return res;
            }
        }


        /**
         * 在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。 你有一辆油箱容量无限的的汽车，从第 i 个加油站 开往第
         * i+1 个加油站需要消耗汽油 cost[i] 升。 你从其中的一个加油站出发，开始时油箱为空。
         * 如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
         *
         * @param gas
         * @param cost
         * @return
         */
        public int canCompleteCircuit(int[] gas, int[] cost) {
            int rest = 0, run = 0, start = 0;
            for (int i = 0; i < gas.length; i++) {
                run += (gas[i] - cost[i]);
                rest += (gas[i] - cost[i]);
                if (run < 0) {
                    start = i + 1;
                    run = 0;
                }
            }
            return rest < 0 ? -1 : start;
        }

        /**
         * 给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。 找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
         * XO 生命游戏
         * @param board
         */
        static int[][] next = { { -1, 0 }, { 1, 0 }, { 0, 1 }, { 0, -1 } };

        public void solve(char[][] board) {
            if (board == null || board.length == 0) {
                return;
            }
            int row = board.length;
            int col = board[0].length;
            for (int i = 0; i < row; i++) {
                backtrace(board, i, 0); // 第一列
                backtrace(board, i, col - 1); // 最后一列
            }
            for (int i = 0; i < col; i++) {
                backtrace(board, 0, i);//第一行
                backtrace(board, row - 1, i);//最后一行
            }
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    if (board[i][j] == 'O')
                        board[i][j] = 'X';
                    if (board[i][j] == '-') {
                        board[i][j] = 'O';
                    }
                }
            }
        }

        private void backtrace(char[][] board, int i, int j) {
            if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || board[i][j] != 'O')
                return;
            board[i][j] = '-';
            for (int[] n : next) {
                backtrace(board, i + n[0], j + n[1]);
            }
            return;
        }

        /**
         * 给定一个未排序的整数数组，找出最长连续序列的长度。要求算法的时间复杂度为 O(n)。
         */
        public static int longedConsecutive(int[] nums) {
            Set<Integer> set = new HashSet<>();
            for (Integer n : nums) {
                set.add(n);
            }
            int longest = 0;
            for (Integer num : nums) {
                if (set.remove(num)) {
                    int curLongest = 1;
                    int cur = num;
                    while (set.remove(cur - 1)) {
                        cur--;
                    }
                    curLongest += (num - cur);
                    cur = num;
                    while (set.remove(cur + 1)) {
                        cur++;
                    }
                    curLongest += (cur - num);
                    longest = Math.max(longest, curLongest);
                }
            }
            return longest;
        }

        /**
         * 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上
         *
         * @param triangle
         * @return
         */
        public static int minimumTotal(List<List<Integer>> triangle) {
            // 两种解法，自顶向下
            // 从底向上
            if (triangle == null || triangle.size() == 0) {
                return 0;
            }

            int[][] dp = new int[triangle.size() + 1][triangle.size() + 1];
            for (int i = triangle.size() - 1; i >= 0; i--) {
                List<Integer> tmp = triangle.get(i);
                for (int j = 0; j < tmp.size(); j++) {
                    dp[i][j] = Math.min(dp[i + 1][j], dp[i + 1][j + 1]) + tmp.get(j);
                }
            }
            return dp[0][0];
        }

        /**
         * 给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
         */
        public static List<List<Integer>> generate(int numRows) {
            List<List<Integer>> result = new ArrayList<>();
            if (numRows == 0) {
                return result;
            }
            int[][] tmp = new int[numRows][numRows];
            for (int i = 0; i < numRows; i++) {
                List<Integer> list = new ArrayList<>();
                for (int j = 0; j <= i; j++) {
                    if (j == 0 || j == i) {
                        tmp[i][j] = 1;
                    } else {
                        tmp[i][j] = tmp[i - 1][j - 1] + tmp[i - 1][j];
                    }
                    list.add(tmp[i][j]);
                }
                result.add(list);
            }
            return result;
        }

        /**
         * 给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。
         */
        public static List<Integer> getRow(int rowIndex) {
            return generate(rowIndex).get(rowIndex);
        }

        /**
         * 直接使用组合公式C(n,i) = n!/(i!*(n-i)!) 则第(i+1)项是第i项的倍数=(n-i)/(i+1);
         */
        public static List<Integer> getRowByMath(int rowIndex) {
            // return generate(rowIndex).get(rowIndex);
            List<Integer> result=new ArrayList<>();
            long cur=1;
            for(int i=0;i<=rowIndex;i++){
                result.add((int)cur);
                cur=cur*(rowIndex-i)/(i+1);
            }
            return result;
        }

        /**
         * 给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
         */
        public static void merge(int[] nums1, int m, int[] nums2, int n) {
            int last = m + n - 1;
            while (n != 0) {
                if (m == 0 || nums1[m - 1] <= nums2[n - 1]) {
                    nums1[last--] = nums2[--n];
                } else {
                    nums1[last--] = nums2[--m];
                }
            }
        }

        public static class SingleStack {
            // 能接住的雨水量
            public static int trap(int[] height) {
                int n = height.length;
                int[] right = new int[height.length];
                int[] left = new int[height.length];
                for (int i = 1; i < n; i++) {
                    left[i] = Math.max(left[i - 1], height[i - 1]);
                    // 每次递增找最大左
                }
                for (int i = n - 2; i >= 0; i--) {
                    // right.add(i,Math.max(right.get(i+1),height[i+1]));
                    right[i] = Math.max(right[i + 1], height[i + 1]);
                    // 找最大右
                }
                int water = 0;
                for (int i = 0; i < n; i++) {
                    // 找出两者最小的 找出上确界 下确界是当前高度 剩下的就是雨水
                    int level = Math.min(left[i], right[i]);

                    water += Math.max(0, level - height[i]);
                }
                return water;

            }

            /**
             * 给定 n 个非负整数表示每个 宽度为 1 的柱子的高度图， 计算按此排列的柱子，下雨之后能接多少雨水。
             */
            public static int trapStack(int[] height) {
                Stack<Integer> stack = new Stack<Integer>();
                int area = 0;
                for (int i = 0; i < height.length; i++) {

                    while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                        int curIndex = stack.pop();
                        while (!stack.isEmpty() && height[curIndex] == height[stack.peek()]) {
                            stack.pop();
                        }
                        if (!stack.isEmpty()) {
                            int h = stack.peek();
                            area += (Math.min(height[h], height[i]) - height[curIndex]) * (i - h - 1);
                        }
                    }

                    stack.push(i);
                }
                return area;

            }

            /**
             * 给定两个 没有重复元素 的数组 nums1 和 nums2 ， 其中nums1 是 nums2 的子集。
             * 找到 nums1 中每个元素在 nums2 中的下一个比其大的值
             *
             * @param nums1
             * @param nums2
             * @return
             */
            public static int[] nextGreaterElement(int[] nums1, int[] nums2) {
                // return null;
                Stack<Integer> stack = new Stack<Integer>();
                HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
                int[] result = new int[nums1.length];
                for (int num : nums2) {
                    while (!stack.isEmpty() && stack.peek() < num) {
                        map.put(stack.pop(), num);
                    }
                    stack.push(num);
                }
                for (int i = 0; i < nums1.length; i++) {
                    result[i] = map.getOrDefault(nums1[i], -1);
                }
                return result;
            }

            /**
             * 请根据每日 气温 列表，重新生成一个列表。 对应位置的输出为：
             * 要想观测到更高的气温，至少需要等待的天数。
             *
             * @param T
             * @return
             */
            public static int[] dailyTemperatures(int[] T) {
                Stack<Integer> stack = new Stack<Integer>();
                int[] result = new int[T.length];
                for (int i = 0; i < T.length; i++) {
                    while (!stack.isEmpty() && T[i] > T[stack.peek()]) {
                        int tmp = stack.pop();
                        result[tmp] = i - tmp;
                    }
                    stack.push(i);
                }
                return result;
            }

            /**
             * 给定一个仅包含 0 和 1 的二维二进制矩阵，
             * 找出只包含 1 的最大矩形，并返回其面积。
             *
             * @param matrix
             * @return
             */
            public int maximalRectangle(char[][] matrix) {
                // return 0;
                if (matrix.length == 0 || matrix[0].length == 0) {
                    return 0;
                }
                int col = matrix.length;
                int row = matrix[0].length;
                int[] height = new int[row];
                int ans = 0;
                for (int i = 0; i < col; i++) {
                    for (int j = 0; j < row; j++) {
                        if (matrix[i][j] == '1') {
                            height[j] += 1;
                        } else {
                            height[j] = 0;
                        }
                    }
                    ans = Math.max(ans, largestRectangleArea(height));
                }
                return ans;

            }

            /**
             * 单调栈
             *
             * @param heights
             * @return
             */
            public static int largestRectangleArea(int[] heights) {
                int[] tmp = new int[heights.length + 2];
                System.arraycopy(heights, 0, tmp, 0, heights.length);
                Deque<Integer> deque = new ArrayDeque<Integer>();
                int area = 0;
                for (int i = 0; i < tmp.length; i++) {
                    while (!deque.isEmpty() && tmp[i] < tmp[deque.peek()]) {
                        int h = tmp[deque.pop()];
                        area = Math.max(area, (i - deque.peek() - 1) * h);
                    }
                    deque.push(i);
                }
                return area;
            }
        }

        /**
         * 旋转数组 寻找是否具有某一个值
         *
         * @param nums
         * @param target
         * @return
         */
        public static boolean searchBinary(int[] nums, int target) {
            int l = 0, r = nums.length - 1;
            while (l <= r) {
                while (l < r && nums[l] == nums[l + 1])
                    ++l;
                while (l < r && nums[r] == nums[l - 1])
                    --r;
                int mid = l + (r - l) / 2;
                if (nums[mid] >= nums[l]) {
                    if (target < nums[mid] && target >= nums[l])
                        r = mid - 1;
                    else
                        l = mid + 1;
                } else {
                    if (target > nums[mid] && target <= nums[r])
                        l = mid + 1;
                    else
                        r = mid - 1;
                }
            }
            return false;
        }

        /**
         * 给定一个排序数组，你需要在原地删除重复出现的元素， 使得每个元素最多出现两次，返回移除后数组的新长度。
         *
         * @param nums
         */
        public static int removeDuplicatesT(int[] nums) {
            int i = 0;
            for (int num : nums) {
                if (i < 2 || num > nums[i - 2]) {
                    nums[i++] = num;
                }
            }
            return i;
        }

        /**
         * 给定一个包含红色、白色和蓝色，一共 n 个元素的数组， 原地对它们进行排序，使得相同颜色的元素相邻， 并按照红色、白色、蓝色顺序排列。
         * 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色
         *
         * @param nums
         */
        public void sortColors(int[] nums) {
            // Arrays.sort(nums);
            int l = -1;
            int r = -1;
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] < 2) {
                    r++;
                    swap(nums, i, r);
                    if (nums[r] < 1) {
                        l++;
                        swap(nums, r, l);
                    }
                }
            }
        }

        private void swap(int[] nums, int i, int j) {
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }

        /**
         * 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。 该矩阵具有如下特性： 每行中的整数从左到右按升序排列。
         * 每行的第一个整数大于前一行的最后一个整数。
         *
         * @param matrix
         * @param target
         * @return
         */
        public boolean searchMatrix(int[][] matrix, int target) {
            if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
                return false;
            }
            int row = matrix.length, col = matrix[0].length;
            int i = 0, j = col - 1;
            while (i <= row - 1 && j >= 0) {
                if (matrix[i][j] == target) {
                    return true;
                } else if (matrix[i][j] > target) {
                    j--;
                } else {
                    i++;
                }
            }
            return false;
        }

        /**
         * 给定一个 m x n 的矩阵，如果一个元素为 0， 则将其所在行和列的所有元素都设为 0。请使用原地算法。
         *
         * @param matrix 替换 然后将代码进行处理
         */
        public void setZeroes(int[][] matrix) {
            Random random = new Random();
            Integer base = random.nextInt(1234567890) + Integer.MAX_VALUE / 100;
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    if (matrix[i][j] == 0) {
                        for (int k = 0; k < matrix.length; k++) {
                            if (matrix[k][j] != 0)
                                matrix[k][j] = base;
                        }
                        for (int k = 0; k < matrix[0].length; k++) {
                            if (matrix[i][k] != 0)
                                matrix[i][k] = base;
                        }
                    }
                }
            }
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[0].length; j++) {
                    if (matrix[i][j] == base) {
                        matrix[i][j] = 0;
                    }
                }
            }
        }

        /**
         * 数字内 两数之和 三数之和 四数之和
         */
        public static class NSum {
            /**
             * 给定一个包含 n 个整数的数组 nums 和一个目标值 target， 判断 nums 中是否存在四个元素 a，b，c 和 d ， 使得 a + b +
             * c + d 的值与 target 相等？ 找出所有满足条件且不重复的四元组。
             *
             * @param nums
             * @param target
             * @return
             */
            public static List<List<Integer>> fourSum(int[] nums, int target) {
                Arrays.sort(nums);
                List<List<Integer>> res = new ArrayList<>();
                for (int i = 0; i < nums.length; i++) {

                    if (i > 0 && nums[i] == nums[i - 1]) {
                        continue;
                    }

                    List<List<Integer>> threeSums = threeSum(nums, i + 1, target - nums[i]);
                    for (List<Integer> list : threeSums) {
                        list.add(nums[i]);
                        res.add(list);
                    }
                }
                return res;
            }

            /**
             * 最接近得三数之和
             *
             * @param nums
             * @param target
             * @return
             */
            public static int threeSumClosest(int[] nums, int target) {
                Arrays.sort(nums);
                int closest = nums[0] + nums[1] + nums[2];
                for (int i = 0; i <= nums.length - 2; i++) {
                    int l = i + 1, r = nums.length - 1;
                    while (l < r) {
                        int threeSum = nums[l] + nums[r] + nums[i];

                    }
                }
                return closest;
            }

            /**
             * 三数之和
             *
             * @param nums
             * @param offset
             * @param  target
             * @return
             */

            public static List<List<Integer>> threeSum(int[] nums, int offset, int target) {
                List<List<Integer>> res = new ArrayList<>();
                for (int i = offset; i < nums.length; i++) {
                    if (i > offset && nums[i] == nums[i - 1]) {
                        continue;
                    }
                    List<List<Integer>> twoSums = twoSum(nums, i + 1, target - nums[i]);
                    for (List<Integer> list : twoSums) {
                        list.add(nums[i]);
                        res.add(list);
                    }
                }
                return res;
            }

            /**
             * 两数之和
             *
             * @param nums
             * @param offset
             * @param  target
             * @return
             */
            public static List<List<Integer>> twoSum(int[] nums, int offset, int target) {
                List<List<Integer>> res = new ArrayList<>();
                int left = offset, right = nums.length - 1;
                while (left < right) {
                    int sum = nums[left] + nums[right];
                    if (sum == target) {
                        List<Integer> tuple = new ArrayList<>();
                        tuple.add(nums[left]);
                        tuple.add(nums[right]);
                        res.add(tuple);
                        while (left < --right && nums[right] == nums[right + 1])
                            ;
                        while (++left < right && nums[left] == nums[left - 1])
                            ;
                    } else if (sum > target) {
                        right--;
                    } else {
                        left++;
                    }
                }
                return res;
            }
        }

        /**
         * 二维矩阵翻转
         * @param matrix
         */
        public static void rotate(int[][] matrix) {
            int i = 0;
            int j = matrix.length - 1;
            while (i <= j) {
                int p1 = i;
                int p2 = j;
                while (p1 != j) {
                    int tmp = matrix[i][p1]; // 保存左上
                    matrix[i][p1] = matrix[p2][i]; // 左上换左下
                    matrix[p2][i] = matrix[j][p2]; // 左下换右下
                    matrix[j][p2] = matrix[p1][j]; // 右下换右上
                    matrix[p1][j] = tmp; // 右上换左上
                    p1++;
                    p2--;
                }
                i++;
                j--;
            }
        }

        /**
         * 给出一个区间的集合，请合并所有重叠的区间。 [[1,6],[8,10],[15,18]] 贪心
         * 合并区间 类似题 leetcode 646 但是本质上是不同的
         * @param intervals
         * @return
         */
        public static int[][] merge(int[][] intervals) {
            if (intervals == null || intervals.length <= 1) {
                return intervals;
            }
            List<int[]> intervalsList = new ArrayList<>();
            Arrays.sort(intervals, new Comparator<int[]>() {

                @Override
                public int compare(int[] o1, int[] o2) {
                    // TODO Auto-generated method stub
                    if (o1[0] == o2[0])
                        return o1[1] - o2[1];
                    return o1[0] - o2[0];
                }
            });
            int i = 0;
            int n = intervals.length;
            while (i < n) {
                int left = intervals[i][0];
                int right = intervals[i][1];
                while (i < n - 1 && right >= intervals[i + 1][0]) {
                    right = Math.max(right, intervals[i + 1][1]);
                    i++;
                }
                intervalsList.add(new int[] { left, right });
                i++;
            }
            return intervalsList.toArray(new int[intervalsList.size()][2]);
        }

        /**
         * 给出一个无重叠的 ，按照区间起始端点排序的区间列表。 在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
         * 思路如上 建立一个新的数组 进行拷贝 保存插入新的数组 newInterval
         */
        public static int[][] insert(int[][] intervals, int[] newInterval) {
            int[][] tmp = new int[intervals.length + 1][2];
            System.arraycopy(intervals, 0, tmp, 0, intervals.length);
            tmp[intervals.length] = newInterval;
            return merge(tmp);

        }

        // 螺旋矩阵
        public static List<Integer> spiralOrder(int[][] matrix) {
            List<Integer> res = new ArrayList<Integer>();
            if (matrix.length == 0 || matrix[0].length == 0) {
                return res;
            }
            int up = 0, down = matrix.length - 1, left = 0, right = matrix[0].length - 1;
            while (up <= down && left <= right) {
                for (int i = left; i <= right; i++) {
                    res.add(matrix[up][i]);
                }
                up++;
                for (int i = up; i <= down; i++) {
                    res.add(matrix[i][right]);
                }
                right--;
                for (int i = right; i >= left && up <= down; i--) {
                    res.add(matrix[down][i]);
                }
                down--;
                for (int i = down; i >= up && left <= right; i--) {
                    res.add(matrix[i][left]);
                }
                left++;
            }
            return res;
        }

        // 螺旋矩阵 II
        /**
         * 给定一个正整数 n，生成一个包含 1 到 n2 所有元素， 且元素按顺时针顺序螺旋排列的正方形矩阵
         *
         * @param n
         * @return
         */
        public int[][] generateMatrix(int n) {
            int[][] res = new int[n][n];
            int c = 1, j = 0;
            while (c <= n * n) {
                for (int i = 0; i < n - j; i++) {
                    res[j][i] = c++;
                }
                for (int i = j + 1; i < n - j; i++) {
                    res[i][n - j - 1] = c++;
                }
                for (int i = n - j - 2; i >= j; i--) {
                    res[n - j - 1][i] = c++;
                }
                for (int i = n - j - 2; i > j; i--) {
                    res[i][j] = c++;
                }
                j++;
            }
            return res;
        }

        /**
         * 给定一个非负整数数组，你�����初位于数组的第一个位置。 数组中的每个元素代表你在该位置**可以跳跃的最大长度**。 nums[i]
         *
         * @param nums
         * @return
         */
        public static boolean canJump(int[] nums) {

            int len = nums.length;
            if (nums.length == 0 || len <= 1) {
                return true;
            }
            int start = nums[0];
            for (int i = 1; i < len - 1; i++) {
                if (i <= start) {// 该位置**可以跳跃的最大长度**
                    start = Math.max(start, nums[i] + i);
                }
            }

            return start >= len - 1;

        }

        // 跳跃游戏
        public static int jump(int[] nums) {
            if (nums.length == 0)
                return 0;
            int reach = 0;
            int nextR = nums[0];
            int step = 0;
            for (int i = 0; i < nums.length; i++) {
                nextR = Math.max(i + nums[i], nextR);
                if (nextR >= nums.length - 1)
                    return (step + 1);
                if (i == reach) {
                    step++;
                    reach = nextR;
                }
            }
            return step;
        }

        /**
         * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
         *
         * @param nums
         * @param target
         * @return
         */
        public static int search(int[] nums, int target) {
            int l = 0;
            int r = nums.length - 1;
            while (l <= r) {
                int mid = l + (r - l) / 2;
                if (nums[mid] == target) {
                    return mid;
                } else if (nums[mid] < nums[r]) {
                    if (nums[mid] < target && target <= nums[r]) {
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                } else {
                    if (nums[l] <= target && target < nums[mid]) {
                        r = mid - 1;
                    } else {
                        l = mid + 1;
                    }
                }
            }
            return -1;
        }

        /**
         * 给定一个按照升序排列的整数数组 nums，和一个目标值 target。 找出给定目标值在数组中的开始位置和结束位置。
         *
         * @param nums
         * @param target
         * @return
         */
        public static int[] searchRange(int[] nums, int target) {
            int[] re = { -1, -1 };
            if (nums == null) {
                return re;
            }
            int l = 0, r = nums.length - 1;
            while (l < r) {
                int mid = l + (r - l) / 2;
                if (nums[mid] == target) {
                    re[0] = mid;
                    if (nums[mid - 1] == target) {
                        re[1] = mid - 1;
                        break;
                    }
                    if (nums[mid + 1] == target) {
                        re[1] = mid + 1;
                        break;
                    }

                } else if (nums[mid] > target) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            Arrays.sort(re);
            return re;
        }

        /**
         * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引
         *
         * @param nums,target
         * @return index
         */
        public static int searchInsert(int[] nums, int target) {
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] >= target) {
                    return i;
                }
            }
            return nums.length;
        }

        /**
         * 判断一个 9x9 的数独是否有效
         *
         * @param board
         * @return
         */
        public static boolean isValidSudoku(char[][] board) {
            // return true;

            boolean[][] row = new boolean[9][9];
            boolean[][] col = new boolean[9][9];
            boolean[][] block = new boolean[9][9];

            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    if (board[i][j] != '.') {
                        int num = board[i][j] - '1';
                        int blockIdx = i / 3 * 3 + j / 3;
                        if (row[i][num] || col[j][num] || block[blockIdx][num]) {
                            return false;
                        } else {
                            row[i][num] = true;
                            row[j][num] = true;
                            block[blockIdx][num] = true;
                        }
                    }
                }
            }
            return false;
        }

        /**
         * 编写一个程序，通过已填充的空格来解决数独问题。 一个数独的解法需遵循如下规则：
         * 数字 1-9 在每一行只能出现一次。 数字 1-9 在每一列只能出现一次。
         * 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
         *
         * @param board char[][]
         */
        public static void solveSudoku(char[][] board) {
            boolean[][] row = new boolean[9][9];
            boolean[][] col = new boolean[9][9];
            boolean[][] block = new boolean[9][9];
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    if (board[i][j] != '.') {
                        int num = board[i][j] = '1';
                        row[i][num] = true;
                        col[j][num] = true;
                        block[i / 3 * 3 + j][num] = true;
                    }
                }
            }
            dfs(board, row, col, block, 0, 0);
        }

        private static boolean dfs(char[][] board, boolean[][] row, boolean[][] col, boolean[][] block, int i, int j) {
            while (board[i][j] != '.') {
                if (++j >= 9) {
                    i++;
                    j = 0;
                }
                if (i >= 9) {
                    return true;
                }
            }
            for (int num = 0; num < 9; num++) {
                int blockIdx = i / 3 * 3 + j / 3;
                if (!row[i][num] && !col[i][num] && !block[blockIdx][num]) {
                    board[i][j] = (char) ('1' + num);
                    row[i][num] = true;
                    col[j][num] = true;
                    block[blockIdx][num] = true;
                    if (dfs(board, row, col, block, i, j)) {
                        return true;
                    } else {
                        row[i][num] = false;
                        col[j][num] = false;
                        block[blockIdx][num] = false;
                        board[i][j] = '.';
                    }
                }
            }
            return false;
        }

        /**
         * 外观数列 「外观数列」是一个整数序列，从数字 1 开始， 序列中的每一项都是对前一项的描述。
         *
         * @param n
         * @return
         */
        public static String countAndSay(int n) {
            if (n == 1)
                return "1";
            String str = "1";
            for (int i = 2; i <= n; i++) {
                StringBuffer sb = new StringBuffer();
                int cnt = 1;
                char pre = str.charAt(0);
                for (int j = 1; j < str.length(); j++) {
                    char back = str.charAt(j);
                    if (back == pre) {
                        cnt++;
                    } else {
                        sb.append(cnt).append(pre);
                        pre = back;
                        cnt = 1;
                    }
                }
                sb.append(cnt).append(pre);
                str = sb.toString();
            }
            return str;
        }

        /**
         * 给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。
         *
         * @param nums
         * @return
         */
        public static int firstMissingPositive(int[] nums) {
            // return 0;
            for (int i = 0; i < nums.length; i++) {
                while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                    swap(nums[i], nums[nums[i] - 1]);
                }
            }
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] != i + 1) {
                    return i + 1;
                }
            }
            return nums.length - 1;
        }

        private static void swap(int i, int j) {
            int t = i;
            i = j;
            j = t;
        }

        /**
         * 给定一个排序数组，你需要在 原地 删除重复出现的元素， 使得每个元素只出现一次，返回移除后数组的新长度。 不要使用额外的数组空间，你必须在 原地
         * 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
         *
         * @param nums
         * @return
         */
        public static int removeDuplicates(int[] nums) {
            int cnt = 0;
            for (int i = 0; i < nums.length; i++) {
                if (nums[cnt] != nums[i]) {
                    nums[++cnt] = nums[i];

                }
            }
            return cnt + 1;
        }

        /**
         *
         * 给你一个数组 nums 和一个值 val， 你需要 原地 移除所有数值等于 val 的元素， 并返回移除后数组的新长度。
         *
         * @param nums
         * @param val
         * @return
         */
        public static int removeElement(int[] nums, int val) {
            int cnt = 0;
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] != val) {
                    nums[cnt] = nums[i];
                    cnt++;
                }
            }
            return cnt;
        }
    }

    public static class BackTrace {

        /**
         * 给定一个仅包含数字 0-9 的字符串和一个目标值，
         * 在数字之间添加二元运算符（不是一元）
         * +、- 或 * ，返回所有能够得到目标值的表达式。
         * 输入: num = "123", target = 6
         * 输出: ["1+2+3", "1*2*3"]
         * @param num
         * @param target
         * @return
         */
        public List<String> addOperators(String num, int target) {
            List<String> res = new ArrayList<>();
            if (num == null || num.length() == 0) {
                return res;
            }

            help(num, target, 0, 0, 0, "", res);
            return res;
        }

        private void help(String num, int target, long curRes, long diff, int start, String curExp, List<String> expressions) {
            if (start == num.length() && (long)target == curRes) {
                expressions.add(new String(curExp));
            }

            for (int i = start; i < num.length(); i++) {
                String cur = num.substring(start, i + 1);
                if (cur.length() > 1 && cur.charAt(0) == '0') {
                    break;
                }

                if (curExp.length() > 0) {
                    help(num, target, curRes + Long.valueOf(cur), Long.valueOf(cur), i + 1, curExp + "+" + cur, expressions);
                    help(num, target, curRes - Long.valueOf(cur), -Long.valueOf(cur), i + 1, curExp + "-" + cur, expressions);
                    help(num, target, (curRes - diff) + diff * Long.valueOf(cur), diff * Long.valueOf(cur),
                          i + 1, curExp + "*" + cur, expressions);
                } else {
                    help(num, target, Long.valueOf(cur), Long.valueOf(cur), i + 1, cur, expressions);
                }
            }
        }
        public static class WordResult {

        //这都是一类题 二维数组内部的回溯算法
        /**
         * 给定一个二维网格和一个单词，找出该单词是否存在于网格中。 单词必须按照字母顺序，通过相邻的单元格内的字母构成，
         * 其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。 同一个单元格内的字母不允许被重复使用。
         *
         * @param board
         * @param word
         * @return
         */
        private final static int[][] next = { { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 } };
        public static boolean exist(char[][] board, String word) {
            int rows = board.length, cols = board[0].length;
            boolean[][] flag = new boolean[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (backtracing(board, flag, 0, i, j, word))
                        return true;
                }
            }
            return false;
        }

        private static boolean backtracing(char[][] board, boolean[][] flag, int pathlen, int i, int j, String word) {
            if (pathlen == word.length()) {
                return true;
            }
            if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || flag[i][j]
                    || board[i][j] != word.charAt(pathlen)) {
                return false;
            }
            flag[i][j] = true;
            for (int[] n : next)
                if (backtracing(board, flag, pathlen + 1, i + n[0], j + n[1], word))
                    return true;
            flag[i][j] = false;
            return false;
        }
        /**
         * 给定一个二维网格 board 和一个字典中的单词列表 words，
         * 找出所有同时在二维网格和字典中出现的单词。
         * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，
         * 其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。
         * 同一个单元格内的字母在一个单词中不允许被重复使用。
         * @param board
         * @param words
         * @return
         */
        private static HashSet<String> wordResults=new HashSet<String>();
        public List<String> findWords(char[][] board, String[] words) {
            boolean[][] flag = new boolean[board.length][board[0].length];
            for(String word: words) {
                for(int i=0; i<board.length; i++) {
                    for(int j=0; j<board[i].length; j++) {
                        backtrace(board,flag,0,i,j,word);
                    }
                }
            }
            return new ArrayList<>(wordResults);
        }
       private  void backtrace(char[][] board,boolean[][] flag,int pathLen,int i,int j,String word) {
            if(pathLen==word.length()){
                wordResults.add(word);
                return;
            }
            if(i<0||j<0||i>=board.length||j>=board[0].length||flag[i][j]|| board[i][j] != word.charAt(pathLen)){
                return ;
            }
            flag[i][j]=true;
            for(int[] n:next){
            backtrace(board, flag, pathLen+1, i+n[0], j+n[1], word);
            }
            flag[i][j]=false;
        }
        /**
         * 给你一个由 '1'（陆地）和 '0'（水）组成的
         * 的二维网格，请你计算网格中岛屿的数量。
         * 岛屿总是被水包围，并且每座岛屿只能由
         * 水平方向或竖直方向上相邻的陆地连接形成。
         * 此外，你可以假设该网格的四条边均被水包围。
         */
        public int numIslands(char[][] grid) {
            int cnt=0;
            for(int i= 0; i < grid.length; i++){
                for(int j= 0; j < grid[i].length; j++){
                    if(grid[i][j]=='1'){
                        cnt++;
                        backtraces(grid,i,j);
                    }
                }
            }
            return cnt;
        }
        private static void backtraces(char[][] grid,int i,int j){
            if(i<0 || j<0 || i>=grid.length||j>=grid[0].length||grid[i][j]=='0'){
                return;
            }
            grid[i][j] ='0';
            for(int[] n:next){
                backtraces(grid,i+n[0], j + n[1]);
            }
        }
        }


        public static class IPProblems {
            /**
             * 给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。 有效的 IP 地址正好由四个整数（每个整数位于 0 到 255
             * 之间组成），整数之间用 '.' 分隔
             */
            private static List<String> result = new ArrayList<String>();

            public static List<String> restoreIpAddresses(String s) {
                if (s == null || s.length() < 4) {
                    return new ArrayList<String>();
                }
                backtrace(s, 0, 0, new StringBuilder());
                return result;
            }

            private static void backtrace(String s,int start,int pos,StringBuilder sb){
                if(pos>4){
                    return;
                }
                if(start==s.length()&&pos==4){
                    result.add(sb.toString().substring(0,sb.length()-1));
                }
                for(int i=start;i<start+3&&i<s.length();i++){
                    String tmp=s.substring(start,i+1);
                    if(Integer.valueOf(tmp)<=255){
                        sb.append(tmp+".");
                        backtrace(s,i+1,pos+1,sb);
                        sb.delete(sb.length()-(i-start+2),sb.length());
                    }
                    if(s.charAt(start)=='0'){
                        break;
                    }
                }
            }
        }

        // n皇后问题 种类
        static boolean[] col = null;
        static boolean[] left = null;
        static boolean[] right = null;
        static List<List<String>> res = new ArrayList<>();

        public List<List<String>> solveNQueens(int n) {
            col = new boolean[n];
            left = new boolean[2 * n - 1];
            right = new boolean[2 * n - 1];
            char[][] board = new char[n][n];
            backtrace(board, 0, n);
            return res;
        }

        private static void backtrace(char[][] board, int i, int n) {
            if (i >= n) {
                List<String> list = new ArrayList<String>();
                for (int j = 0; i < n; j++) {
                    list.add(new String(board[j]));
                }
                res.add(list);
                return;
            }
            Arrays.fill(board[i], '.');
            for (int k = 0; k < n; k++) {
                if (!col[k] && !left[k] && !right[i - k + n - 1]) {
                    board[i][k] = 'Q';
                    col[k] = true;
                    left[i + k] = true;
                    backtrace(board, k + 1, n);
                    board[i][k] = '.';
                    col[k] = false;
                    left[i + k] = false;
                    right[i - k + n - 1] = false;
                }
            }
        }

        // n皇后问题 数量
        private static int cnt = 0;

        public static int totalNQueens(int n) {
            boolean[] col = new boolean[n];
            char[][] arr = new char[n][n];
            backtrace(arr, col, 0, n);
            return cnt;
        }

        public static void backtrace(char[][] arr, boolean[] col, int r, int n) {
            if (r >= n) {
                cnt++;
                return;
            }
            for (int i = 0; i < n; i++) {
                boolean sign = false;
                if (col[i])
                    continue;
                for (int a = r - 1, b = i - 1; a >= 0 && b >= 0; a--, b--) {
                    if (arr[a][b] == 'Q') {
                        sign = true;
                        break;
                    }
                }
                if (sign)
                    continue;
                for (int a = r - 1, b = i + 1; a >= 0 && b < n; a--, b++) {
                    if (arr[a][b] == 'Q') {
                        sign = true;
                        break;
                    }
                }
                if (sign)
                    continue;
                col[i] = true;
                arr[r][i] = 'Q';
                backtrace(arr, col, r + 1, n);
                col[i] = false;
                arr[r][i] = 0;
            }
        }

        /**
         * 给定一组不含重复元素的整数数组 nums， 返回该数组所有可能的子集（幂集）。
         *
         * @param nums
         * @return
         */
        public static List<List<Integer>> subsets(int[] nums) {
            List<List<Integer>> result = new ArrayList<List<Integer>>((int) Math.pow(2, nums.length));
            result.add(new ArrayList<>());
            for (int num : nums) {
                int size = result.size();
                for (int i = 0; i < size; i++) {
                    List<Integer> list = new ArrayList<>(result.get(i));
                    list.add(num);
                    result.add(list);
                }
            }
            return result;

        }

        public List<List<Integer>> subsetsWithDup(int[] nums) {
            List<List<Integer>> result = new ArrayList<List<Integer>>();
            Arrays.sort(nums);
            result.add(new ArrayList<>());
            int len = 1;
            List<Integer> tmp = new ArrayList<>();
            tmp.add(nums[0]);
            result.add(tmp);
            if (nums.length == 1)
                return result;
            for (int i = 1; i < nums.length; i++) {
                int size = result.size();
                if (nums[i] != nums[i - 1]) {
                    len = size;
                }
                for (int j = size - len; j < size; j++) {
                    List<Integer> list = new ArrayList<Integer>(result.get(j));
                    list.add(nums[i]);
                    result.add(list);
                }
            }

            return result;
        }

        public static List<List<Integer>> subsetsBack(int[] nums) {
            List<List<Integer>> result = new ArrayList<List<Integer>>((int) Math.pow(2, nums.length));
            result.add(new ArrayList<>());

            for (int i = 0; i < nums.length; i++) {
                backtrace(result, new ArrayList<>(), i, 0, nums);
            }
            return result;
        }

        private static void backtrace(List<List<Integer>> result, List<Integer> tmp, int i, int j, int[] nums) {
            if (tmp.size() - 1 >= i) {
                result.add(new ArrayList<Integer>(tmp));
                // Java 容器添加为浅复制， C++ 容器添加为深复制
                // 若改成 ans.add(tmp),
                // 则 ans 所有的元素都是 tmp 的浅复制，
                // 最后都为 []
                return;
            }
            for (int m = j; m < nums.length; m++) {
                tmp.add(nums[m]);
                m++;
                backtrace(result, tmp, i, m, nums);
                m--;
                tmp.remove(tmp.size() - 1);
            }
        }

        private static List<List<Integer>> resultofDup = new ArrayList<>();
        private static List<Integer> lsitofDup = new ArrayList<>();

        public List<List<Integer>> subsetsWithDupBack(int[] nums) {
            if (nums.length == 0)
                return resultofDup;
            Arrays.sort(nums);
            backtraceDup(nums, 0);
            return result;
        }

        private static void backtraceDup(int[] nums, int i) {
            resultofDup.add(lsitofDup);

            for (int j = i; j < nums.length; i++) {
                if (j > i && nums[j] == nums[j - 1])
                    continue;
                lsitofDup.add(nums[j]);
                backtrace(nums, j + 1);
                lsitofDup.remove(lsitofDup.size() - 1);
            }
        }

        private static List<List<Integer>> results = new ArrayList<List<Integer>>();
        private static List<Integer> tmps = new ArrayList<Integer>();

        public static List<List<Integer>> subsetsBackII(int[] nums) {
            backtrace(nums, 0);
            return result;
        }

        private static void backtrace(int[] nums, int i) {
            if (i >= nums.length) {
                results.add(new ArrayList<>(tmps));
                return;
            }
            backtrace(nums, i + 1);
            tmps.add(nums[i]);
            backtrace(nums, i + 1);
            tmps.remove(tmps.size() - 1);
        }



        // 组合总和
        /**
         * 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
         *
         * @param n
         * @param k
         * @return
         */
        private static List<List<Integer>> result;
        private static List<Integer> tmp;

        public static List<List<Integer>> combine(int n, int k) {
            result = new ArrayList<>();
            tmp = new ArrayList<>();
            backtrace(k, 0, n);
            return result;
        }

        private static void backtrace(int remain, int last, int n) {
            if (remain == 0) {
                result.add(new ArrayList<>(tmp));
                return;
            }
            for (int i = last + 1; i <= n; i++) {
                tmp.add(i);
                backtrace(remain - 1, i, n);
                tmp.remove(tmp.size() - 1);
            }
        }

        public List<List<Integer>> combinationSum3(int k, int n) {
            List<List<Integer>> result = new ArrayList<List<Integer>>();
            List<Integer> tmp=new ArrayList<Integer>();
            backtrace(n,result,k,tmp,0);
            return result;
        }
        private void backtrace(int n,List<List<Integer>> result, int k, List<Integer> tmp, int i){
            if(n<0){
                return ;
            }
            if(k==0){
                if(n==0){
                    result.add(new ArrayList<>(tmp));
                }
                return;
            }
            for(int j=i+1;j<=9;j++){
                tmp.add(j);
                backtrace(n-j,result,k-1,tmp,j);
                tmp.remove(tmp.size()-1);
            }
            return ;
        }
        /**
         * 给定一个无重复元素的数组 candidates 和一个目标数 target ， 找出 candidates 中所有可以使数字和为 target 的组合。
         * candidates 中的数字可以无限制重复被选取。
         * @param candidates
         * @param target
         * @return
         */
        public static List<List<Integer>> combinationSum(int[] candidates, int target) {
            // return null;
            List<List<Integer>> result = new ArrayList<List<Integer>>();
            Arrays.sort(candidates);
            backtracing(candidates, target, result, 0, new ArrayList<>());
            return result;
        }

        private static void backtracing(int[] candidates, int target, List<List<Integer>> result, int i,
                List<Integer> tmp) {
            if (target < 0)
                return;
            if (target == 0) {
                result.add(tmp);
                return;
            }
            for (int j = i; j < candidates.length; j++) {
                if (target < 0)
                    return;
                tmp.add(candidates[j]);
                backtracing(candidates, target - candidates[j], result, j, tmp);
                tmp.remove(tmp.size() - 1);
            }
        }

        /**
         * 给定一个数组 candidates 和一个目标数 target ， 找出 candidates 中所有可以使数字和为 target 的组合。
         *
         * @param candidates
         * @param target
         * @return
         */
        public static List<List<Integer>> combinationSum2(int[] candidates, int target) {
            List<List<Integer>> result = new ArrayList<>();
            Arrays.sort(candidates);
            backtrace(candidates, target, result, 0, new ArrayList<>());
            return result;
        }

        private static void backtrace(int[] candidates, int target, List<List<Integer>> result, int i,
                List<Integer> tmp) {
            if (target < 0)
                return;
            if (target == 0) {
                result.add(tmp);
                return;
            }
            for (int j = i; j < candidates.length; j++) {
                if (target < 0)
                    return;
                if (j > i && candidates[j] == candidates[j - 1])
                    continue;
                tmp.add(candidates[j]);
                backtracing(candidates, target - candidates[j], result, j, tmp);
                tmp.remove(tmp.size() - 1);
            }
        }

        /**
         * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
         *
         * @param digits
         * @return
         */
        private static HashMap<Character, String> map = new HashMap<>();
        private static List<String> list = new ArrayList<>();

        public static List<String> letterCombinations(String digits) {
            if (digits == null) {
                return null;
            }
            map.put('2', "abc");
            map.put('3', "def");
            map.put('4', "ghi");
            map.put('5', "jkl");
            map.put('6', "mno");
            map.put('7', "pqrs");
            map.put('8', "tuv");
            map.put('9', "wxyz");
            backtrace(digits, 0, new StringBuffer());
            return list;
        }

        private static void backtrace(String digits, int l, StringBuffer sb) {
            if (l == digits.length()) {
                list.add(sb.toString());
                return;
            }
            char ch = digits.charAt(l);
            String s = map.get(ch);
            for (int i = 0; i < s.length(); i++) {
                sb.append(s.charAt(i));
                backtrace(digits, l + 1, sb);
                sb.deleteCharAt(sb.length() - 1);
            }
        }
        /**
         * 1. 字符串的排列
         * 题目描述 输入一个字符串，按字典序打印出该字符串中字符的所有排列。例如输入字符串 abc，
         * 则打印出由字符 a, b, c 所能排列出来的所有字符串 abc, acb, bac, bca, cab 和 cba。
         */
        private List<String> permunate=new ArrayList<>();
        public List<String> permunation(String str){
            if(str==null) {
                return permunate;
            }
            char[] ch=str.toCharArray();
            Arrays.sort(ch);
            backtrace(ch,new boolean[ch.length],new StringBuilder());
            return permunate;
        }
        //dfs 创建一个boolean 数组 判断这个有没有用过
        private void backtrace(char[] ch,boolean[] flag,StringBuilder sb){
            if(sb.length()==ch.length){
                permunate.add(sb.toString());
                return;
            }
            for(int i=0;i<ch.length;i++){
                if(flag[i]){
                    continue;
                }
                if(i!=0&&ch[i]==ch[i-1]&&!flag[i-1]){
                    continue;
                }
                flag[i]=true;
                sb.append(ch[i]);
                backtrace(ch,flag,sb);
                sb.deleteCharAt(sb.length()-1);
                flag[i]=false;
            }


        }
        // 给定一个 没有重复 数字的序列，返回其所有可能的全排列。
        public static List<List<Integer>> permute(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            int[] isVisited = new int[nums.length];
            backtrack(result, nums, new ArrayList<>(), isVisited);
            return result;
        }

        private static void backtrack(List<List<Integer>> result, int[] nums, List<Integer> tmp, int[] isVisited) {
            if (tmp.size() == nums.length) {
                result.add(tmp);
                return;
            }

            for (int i = 0; i < nums.length; i++) {
                if (isVisited[i] == 1)
                    continue;
                isVisited[i] = 1;
                tmp.add(nums[i]);
                backtrack(result, nums, tmp, isVisited);
                isVisited[i] = 0;
                tmp.remove(tmp.size() - 1);
            }
        }

        public static List<List<Integer>> permuteUnique(int[] nums) {
            List<List<Integer>> result = new ArrayList<>();
            int[] isVisited = new int[nums.length];
            backtracks(result, nums, new ArrayList<>(), isVisited);
            return result;
        }

        private static void backtracks(List<List<Integer>> result, int[] nums, List<Integer> tmp, int[] isVisited) {
            if (tmp.size() == nums.length) {
                result.add(tmp);
                return;
            }

            for (int i = 0; i < nums.length; i++) {
                if (i > 0 && nums[i] == nums[i - 1] || isVisited[i] == 1)
                    continue;
                isVisited[i] = 1;
                tmp.add(nums[i]);
                backtrack(result, nums, tmp, isVisited);
                isVisited[i] = 0;
                tmp.remove(tmp.size() - 1);
            }
        }

        /**
         * 给定一个字符串 s1，我们可以把它递归地分割成两个非空子字符串 ，从而将其表示为二叉树。 s1 和 s2是否是扰乱字符串
         *
         * @param s1
         * @param s2
         * @return
         */
        public static boolean isScramble(String s1, String s2) {
            if (s1.length() != s2.length())
                return false;

            return dfs(s1, s2);
        }

        private static boolean dfs(String s1, String s2) {
            char[] ch = s1.toCharArray();
            char[] ch2 = s2.toCharArray();
            Arrays.sort(ch);
            Arrays.sort(ch2);
            if (!String.valueOf(ch).equals(String.valueOf(ch2)))
                return false=;
            int n = s1.length();
            if (n == 1)
                return true;
            for (int i = 0; i < n; i++) {
                if (dfs(s1.substring(0, i), s2.substring(0, i)) && dfs(s1.substring(i, n), s2.substring(i, n))) {
                    return true;
                }
                if (dfs(s1.substring(0, i), s2.substring(n - i, n))
                        && dfs(s1.substring(i, n), s2.substring(0, n - i))) {
                    return true;
                }
            }
            return false;
        }
    }


    public static class StringOperations {
        /**
         * 句子翻转
         * @param str
         * @return
         */
        public String ReverseSentence(String str) {
            String[] tmp=str.trim().split(" ");
            List<String> strTmp=new ArrayList<>();
            strTmp=Arrays.asList(tmp);
            Collections.reverse(strTmp);
            return strTmp.toString();
        }
        /**
         * 给定一个单词，你需要判断单词的大写使用是否正确。
         *
         * 我们定义，在以下情况时，单词的大写用法是正确的：
         *
         * 全部字母都是大写，比如"USA"。
         * 单词中所有字母都不是大写，比如"leetcode"。
         * 如果单词不只含有一个字母，只有首字母大写， 比如 "Google"。
         * 否则，我们定义这个单词没有正确使用大写字母。
         */
        public boolean detectCapitalUse(String word) {
            char[] ch=word.toCharArray();
            int uper=0,lower=0;
            for(char c:ch){
                if(c-'a'>=0){
                    lower++;
                }else{
                    uper++;
                }
            }
            if(uper==ch.length){
                return true;
            }
            if(lower==ch.length){
                return true;
            }
            if(ch[0]<='Z'&&ch[0]>='A'&&uper==1&&lower==ch.length-1){
                return true;
            }
            return false;
        }
        /**
         * 字符串压缩
         * @param S
         * @return
         */
        public String compressString(String S) {
            char[] ch=S.toCharArray();
            StringBuilder sb=new StringBuilder();
            int cnt=1;
            for(int i=0;i<ch.length;i++){
                sb.append(ch[i]);
                while(i+1<ch.length&&ch[i]==ch[i+1]){
                    i++;
                    cnt++;
                }
                sb.append(cnt);
                cnt=1;
            }
            return sb.length()>= S.length()? S:sb.toString();
        }
        /**
         * 给定一个整数，将其转化为7进制，并以字符串形式输出。
         * 36进制 需要分为0-9 A-Z 进行考虑
         * 有点像x这道题
         * @param nums
         * @return
         */
        public String convertToBase7(int nums) {
            StringBuilder sb=new StringBuilder();
            boolean flag=false;
            if(nums==0){
                return "0";
            }
            if(nums<0){
                flag=true;
                nums=-nums;
            }
            while(nums>0){
                sb.append((char)(nums%7+'0'));
                nums/=7;
            }
            if(flag){
                sb.append("-");
            }
            return sb.reverse().toString();
        }
        /**
         * 将非负整数转换为其对应的英文表示。可以保证给定输入小于 2^31 - 1
         * @param num
         * @return
         */
        String[] low={"","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"};
        String[] mid={"Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"};
        String[] high={"","","Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"};
        public String numberToWords(int num) {
            if(num==0){
                return "Zero";
            }
            int part1=num%1000;
            num/=1000;
            int part2=num%1000;
            num/=1000;
            int part3=num%1000;
            num/=1000;
            int part4=num;
            String ans="";
            if(part4!=0){
                ans+=build(part4)+"Billion ";
            }
            if(part3!=0){
                ans+=build(part3)+"Million ";
            }
            if(part2!=0){
                ans+=build(part2)+"Thousand ";
            }
            if(part1!=0){
                ans+=build(part1);
            }
            return ans.trim();
        }
        private String build(int num){
            int a=num%10;
            num/=10;
            int b=num%10;
            num/=10;
            int c=num;
            String ans="";
            if(c!=0){
                ans+=low[c]+" Hundred ";
            }
            if(b==1){
                ans+=mid[a];
            }else if(b==0){
                ans+=low[a];
            }else{
                ans+=high[b]+" "+low[a];
            }
            return ans.trim()+" ";
        }
        public boolean isAnagram(String s, String t) {
            char[] a = s.toCharArray();
            Arrays.sort(a);
            char[] b = t.toCharArray();
            Arrays.sort(b);
            return String.valueOf(a).equals(String.valueOf(b)); //
        }
        /**
         * 给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。
         * 你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。
         * 1+2*3 => (1+2)*3
         * @param input
         */
        private HashMap<String,List<Integer>> hashmap=new HashMap<>();
        public List<Integer> diffWaysToCompute(String input) {
            if(hashmap.containsKey(input)) return hashmap.get(input);
            List<Integer> list = new ArrayList<>();

            int len=input.length();
            for(int i=0; i<len; i++){
                char c=input.charAt(i);
                if(c=='+'||c=='-' || c=='*'){
                    List<Integer> left=diffWaysToCompute(input.substring(0, i));
                    List<Integer> right=diffWaysToCompute(input.substring(i+1, len));

                    for(int l:left){
                        for(int r:right){
                            switch(c){
                                case '+':
                                    list.add(l+r);
                                    break;
                                case '-':
                                    list.add(l-r);
                                    break;
                                case '*':
                                    list.add(l*r);
                                    break;
                            }
                        }
                    }
                }
            }
            if(list.size()==0) list.add(Integer.valueOf(input));
            hashmap.put(input,list);
            return list;
        }
        static  class Palidrome{
            /**
             * 分割回文子串
             *
             * @param s
             * @return
             */
            public List<List<String>> partition(String s) {
                List<List<String>> list = new ArrayList<List<String>>();
                backtrace(s, 0, new ArrayList<>(), list);
                return list;
            }

            private static void backtrace(String s, int offset, List<String> tmp, List<List<String>> result) {
                if (offset == s.length()) {
                    result.add(new ArrayList<>(tmp));
                    return;
                }
                for (int i = offset; i < s.length(); i++) {
                    String tempString = s.substring(offset, i + 1);
                    if (!isPalindromes(tempString)) {
                        continue;
                    }
                    tmp.add(tempString);
                    backtrace(s, offset + 1, tmp, result);
                    tmp.remove(tmp.size() - 1);
                }
            }

            private static boolean isPalindromes(String s) {
                if (s == null || s.length() <= 1) {
                    return true;
                }
                int left = 0;
                int right = s.length() - 1;
                while (left < right) {
                    if (s.charAt(left) != s.charAt(right)) {
                        return false;
                    }
                    left++;
                    right--;
                }
                return true;
            }

            public boolean isPalindrome(String s) {
                String tmp = s.toLowerCase();
                StringBuffer sb = new StringBuffer();
                for (char c : tmp.toCharArray()) {
                    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')) {
                        sb.append(c);
                    }
                }
                return sb.reverse().toString().equals(sb.toString());
            }
            public int longestPalindrome(String s) {
                int[] ch=new int[256];
                for(char c:s.toCharArray()){
                    ch[c-'A']++;
                }
                int cnt=0;
                for(int i:ch){
                    cnt+=i-(i&1);
                }
                return cnt<s.length()?cnt+1:cnt;
            }

            /**
             * 给定一个字符串 s，你可以通过在字符串前面
             * 添加字符将其转换为回文串。
             * 找到并返回可以用这种方式转换的最短回文串。
             * @param s
             * @return
             */
            public String shortestPalindrome(String s) {
                StringBuilder sb = new StringBuilder(s).reverse();
                String str=s+"#"+sb;
                int[] next=next(str);
                String result=sb.substring(0,sb.length()-next[str.length()]);
                return result;

            }
            private int[] next(String str) {
                int[] next=new int[str.length()+1];
                next[0]=-1;
                int k=-1;
                int i=1;
                while(i<next.length) {
                    if(k==-1||str.charAt(k)==str.charAt(i-1)){
                        next[i++]=++k;
                    }else{
                        k=next[k];
                    }
                }
                return next;
            }
        }

        /**
         * 所有 DNA 都由一系列缩写为 A，C，G 和 T 的
         * 核苷酸组成，例如：“ACGAATTCCG”。在研究 DNA 时，
         * 识别 DNA 中的重复序列有时会对研究非常有帮助。
         * 编写一个函数来查找目标子串，目标子串的长度为 10，
         * 且在 DNA 字符串 s 中出现次数超过一次。
         * @param s
         * @return
         */
        public List<String> findRepeatedDnaSequences(String s) {
            HashSet<String> result = new HashSet<String>();
            HashSet<String> set = new HashSet<String>();
            for(int i=0; i<=s.length()-10; i++){
                String tmp=s.substring(i,i+10);
                if(!set.add(tmp)) result.add(tmp);
            }
            return new ArrayList<String>(result);
        }
         /**
         * 数组内数字组合成为最大的数
         * @param nums
         * @return
         */
        public String largestNumber(int[] nums){
            return Arrays.stream(nums)
            .boxed()
            .map(o-> Integer.toString(o)).sorted(
                (s1, s2) ->{
                    String sum1=s1+s2;
                    String sum2=s2+s1;
                    for(int i=0; i<sum1.length(); i++){
                        if(sum1.charAt(i)!=sum2.charAt(i)){
                            return sum2.charAt(i) - sum1.charAt(i);
                        }
                    }
                    return 0;
                }
            ).reduce(String::concat)
            .filter(s->!s.startsWith("0"))
            .orElse("0");
        }

        /**
         *
         * @param n
         * @return
         */
        public String convertToTitle(int n) {
            if(n<0){
                return new String();
            }
            StringBuffer sb=new StringBuffer();
            while(n>0){
                n--;
                sb.append((char)(n%26+'A'));
                n/=26;
            }
            return sb.reverse().toString();
        }
        /**
         * 给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以字符串形式返回小数。
         * 如果小数部分为循环小数，则将循环的部分括在括号内。
         * @param numerator
         * @param denominator
         * @return
         */
        public static String fractionToDecimal(int numerator, int denominator) {
            if (numerator == 0 || denominator == 0) return "0";
            int sign = 1;
            if (numerator > 0 && denominator < 0) sign = -1;
            long big = (long) numerator / (long) denominator;
            long small = numerator % denominator;
            StringBuilder result = new StringBuilder(String.valueOf(big));
            if (sign == -1) result.insert(0, "-");
            if (small != 0) {
                result.append(".");
                StringBuilder smallStr = new StringBuilder();
                Map<String, Integer> smallIndexs = new HashMap<String, Integer>();
                while (small != 0) {
                    small *= 10;
                    big = small / denominator;
                    small = small % denominator;
                    String str = small + "_" + big;
                    if (smallIndexs.containsKey(str)) {
                        smallStr.append(")");
                        smallStr.insert(smallIndexs.get(str), "(");
                        break;
                    } else {
                        smallIndexs.put(str, smallStr.length());
                        smallStr.append(Math.abs(big));
                    }
                }
                result.append(smallStr);
            }
            return result.toString();
        }
        /**
         * 比较两个版本号 version1 和 version2。
         * 如果 version1 > version2 返回 1
         * 此外 返回0
         * @param version1
         * @param version2
         * @return
         */
        public int compareVersion(String version1, String version2) {
            String[] v1=version1.split("\\.");
            String[] v2=version2.split("\\.");
            for(int n=0;n<Math.max(v1.length, v2.length);n++){
                int i=n<v1.length?Integer.valueOf(v1[n]):0;
                int j=n<v2.length?Integer.valueOf(v2[n]):0;
                if(i>j) return 1;
                else if(i<j) return -1;
            }
            return 0;
        }
        /**
         * 给定一个字符串，逐个翻转字符串中的每个单词。
         */
        public String reverseWords(String s) {
            String[] arr=s.trim().split(" +");
            Collections.reverse(Arrays.asList(arr));
            return String.join(" ", arr);
        }
        private static HashMap<String, List<String>> map=new HashMap<>();
        public static List<String> wordBreakList(String s, List<String> wordDict) {
            return backtrace(s,wordDict,0);
        }
        private static List<String> backtrace(String s,List<String> wordDict,int offset){
            //终止条件
            if(offset==s.length()){
                List<String> res=new ArrayList<>();
                res.add("");
                return res;
            }
            if(map.containsKey(s.substring(offset))){
                return map.get(s.substring(offset));
            }
            List<String> res=new ArrayList<>();
            for(String word:wordDict){
                if(word.equals(s.substring(offset,Math.min(s.length(),offset+word.length())))){
                    List<String> tmp=backtrace(s, wordDict, offset+word.length());
                    for(String t:tmp){
                        res.add((word+" "+t).trim());
                    }
                }
            }
            map.put(s.substring(offset), res);
            return res;
        }
        public boolean wordBreak(String s, List<String> wordDict) {
            int n=s.length();
            boolean[] dp=new boolean[n+1];
            dp[0]=true;
            for(int i=1;i<=n;i++){
                for(int j=0;j<i;j++){
                    if(dp[j]&&wordDict.contains(s.substring(j,i))){
                        dp[i]=true;
                        break;
                    }
                }
            }
            return dp[n];
        }
        public static class Ladders {
            /**
             * beginWord = "hit", endWord = "cog", wordList =
             * ["hot","dot","dog","lot","log","cog"] dfs +bfs 无向图
             */
            public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
                return null;
            }

            /**
             * 给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度
             */
            public int ladderLength(String beginWord, String endWord, List<String> wordList) {
                return 0;
            }
        }

        /**
         * 给定一个字符串 s，将 s 分割成一些子串， 使每个子串都是回文串。返回符合要求的最少分割次数。
         *
         * @param s
         * @return
         */
        public int minCut(String s) {
            if (s == null || s.length() <= 1) {
                return 0;
            }
            int len = s.length();
            int[] dp = new int[len];
            for (int i = 0; i < len; i++) {
                backtrack(s, i, i, dp);// 奇数
                backtrack(s, i, i + 1, dp);// 偶数
            }
            return dp[len - 1];
        }

        private void backtrack(String s, int i, int j, int[] dp) {
            int len = s.length();
            while (i >= 0 && j < len && s.charAt(i) == s.charAt(j)) {
                dp[j] = Math.min(dp[j], (i == 0 ? -1 : dp[i - 1]) + 1);
                i--;
                j++;
            }
        }



        /**
         * 给定一个字符串 S 和一个字符串 T，计算在 S 的子序列中 T 出现的个数。 一个字符串的一个子序列是指，通过删除一些（也可以不删除）
         * 字符且不干扰剩余字符相对位置所组成的新字符串
         */
        public static int numDistinct(String s, String t) {
            int m = s.length(), n = t.length();
            int[][] dp = new int[m + 1][n + 1];
            for (int i = 0; i <= m; i++) {
                dp[i][0] = 1;
            }
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= n; j++) {
                    if (j > i) {
                        continue;
                    }
                    if (s.charAt(i - 1) == t.charAt(j - 1)) {
                        dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                    } else {
                        dp[i][j] = dp[i - 1][j];
                    }
                }
            }
            return dp[m][n];
        }

        /**
         * 一条包含字母 A-Z 的消息通过以下方式进行了编码： 给定一个只包含数字的非空字符串，请计算解码方法的总数
         * 输入: "12
         * 输出: 2
         * @param s
         */
        public static int numDecodings(String s) {
            /**
             * 上楼梯的复杂版? 如果连续的两位数符合条件，就相当于一个上楼梯的题目， 可以有两种选法： 1.一位数决定一个字母 2.两位数决定一个字母
             * 就相当于dp(i) = dp[i-1] + dp[i-2]; 如果不符合条件，又有两种情况 1.当前数字是0： 不好意思，这阶楼梯不能单独走， dp[i]
             * = dp[i-2] 2.当前数字不是0 不好意思，这阶楼梯太宽，走两步容易扯着步子， 只能一个一个走 dp[i] = dp[i-1];
             */
            final int length = s.length();
            if (length == 0)
                return 0;
            if (s.charAt(0) == '0')
                return 0;
            int[] dp = new int[length + 1];
            dp[0] = 1;
            for (int i = 0; i < length; i++) {
                dp[i + 1] = s.charAt(i) == '0' ? 0 : dp[i];
                if (i > 0 && (s.charAt(i - 1) == '1' || (s.charAt(i - 1) == '2' && s.charAt(i) <= '6'))) {
                    dp[i + 1] += dp[i - 1];
                }
            }
            return dp[length];
            /**
             * if(s.charAt(0)=='0') return false; int[] dp=new int[s.length()]; dp[0]=1;
             * for(int i=1;i<dp.length;i++){ int cur=s.charAt(i)-'0',pre=s.charAt(i-1)-'0';
             * if(cur==0&&pre!=1&&pre!=2){ return 0; } if(cur!=0){ dp[i]+=dp[i-1]; }
             * if(pre==2&&cur<=6||pre==1){ dp[i]+=i>1?dp[i-2]:1;
             *
             * } return dp[dp.length-1]; }
             */
        }

        /**
         * 给你一个字符串 S、一个字符串 T， 请在字符串 S 里面找出：包含 T 所有字符的最小子串。
         *
         * @param s
         * @param t
         * @return
         */
        public static String minWindow(String s, String t) {
            int[] dp = new int[256];
            for (char c : t.toCharArray())
                dp[c] += 1;
            int start = 0, end = 0;
            int n = s.length(), m = t.length();
            int cnt = 0;
            int res = -1;
            String re = "";
            while (end < n) {
                char c = s.charAt(end);
                dp[c] -= 1;
                if (dp[c] >= 0) {
                    cnt += 1;
                }
                while (cnt == m) {
                    if (res == -1 || res > end - start + 1) {
                        re = s.substring(start, end + 1);
                        res = end - start + 1;
                    }
                    c = s.charAt(start);
                    dp[c] += 1;
                    if (dp[c] >= 1) {
                        cnt -= 1;
                    }
                    start += 1;
                }
                end += 1;
            }
            return re;
        }

        /**
         * 以 Unix 风格给出一个文件的绝对路径， 你需要简化它。或者换句话说，将其转换为规范路径。 "/home/" -> "/home" 做法： 栈
         *
         * @param path
         * @return
         */
        public static String simplifyPath(String path) {
            path = path.replaceAll("/+", "/");
            String[] paths = path.split("/");
            List<String> parts = new ArrayList<String>();
            for (String s : paths) {
                if (s.equals("")) {
                    continue;
                }
                if (s.equals(".")) {
                    continue;
                }
                if (s.equals("..")) {
                    if (parts.size() == 0) {
                        continue;
                    }
                    parts.remove(parts.size() - 1);
                    continue;
                }
                parts.add("/" + s);
            }
            String re = "";
            for (String r : parts) {
                re += r;
            }
            if (re.equals("")) {
                return "/";
            }
            return re;
        }

        /**
         * 给定一个仅包含大小写字母和空格' '的字符串 s， 返回其最后一个单词的长度。 如果字符串从左向右滚动显示， 那么最后一个单词就是最后出现的单词。
         *
         * @param s
         * @return
         */
        public static int lengthOfLastWord(String s) {
            if (s == null) {
                return 0;
            }
            String[] tmp = s.split(" ");
            if (tmp.length == 0) {
                return 0;
            }
            return tmp[tmp.length - 1].length();
        }

        // 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
        public static List<List<String>> groupAnagrams(String[] strs) {
            HashMap<String, ArrayList<String>> map = new HashMap<>();
            for (String str : strs) {
                char[] ch = str.toCharArray();
                Arrays.sort(ch);
                String key = String.valueOf(ch);
                if (!map.containsKey(key)) {
                    map.put(key, new ArrayList<String>());
                }
                map.get(key).add(str);
            }
            return new ArrayList<>(map.values());
        }

        /**
         * 实现 strStr() 函数。给定一个 haystack 字符串和一个 needle 字符串， 在 haystack 字符串中找出 needle
         * 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
         *
         * @param haystack
         * @param needle
         * @return
         */
        public static int strStr(String haystack, String needle) {
            // return haystack.indexOf(needle);
            if (haystack.length() == 0) {
                return 0;
            }
            if (needle.length() > haystack.length()) {
                return -1;
            }
            int i = 0, j = 0;
            while (i < haystack.length() && j < needle.length()) {
                if (j == needle.length()) {
                    return i - needle.length();
                }
                if (haystack.charAt(i) == needle.charAt(j)) {
                    i++;
                    j++;
                } else {
                    i -= j;
                    j = 0;
                }
                if (j == needle.length()) {
                    return i - needle.length();
                }
            }
            return -1;
        }

        /**
         * 两数相除 给定两个整数，被除数 dividend 和除数 divisor。将两数相除， 要求不使用乘法、除法和 mod 运算符。
         * 返回被除数 dividend 除以除数 divisor 得到的商。
         *
         * @param dividend
         * @param divisor
         * @return
         */
        // 未考虑边际条件
        public static int divide(int dividend, int divisor) {
            if (dividend == 0) {
                return 0;
            }
            if (dividend == Integer.MIN_VALUE && divisor == -1) {
                return Integer.MAX_VALUE;
            }
            long tmp_d = Math.abs((long) dividend);
            long tmp_dr = Math.abs((long) divisor);
            int cnt = 0;
            for (int i = 31; i >= 0; i--) {
                if ((tmp_d >> i) >= tmp_dr) {
                    cnt += 1 << i;
                    tmp_d -= tmp_dr << i;
                }
            }
            boolean isNeg = (dividend ^ divisor) < 0;
            return isNeg ? -cnt : cnt;
        }

        /**
         * 给定一个字符串 s 和一些长度相同的单词 words。 找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。
         *
         * @param s
         * @param words
         * @return
         */
        public static List<Integer> findSubstring(String s, String[] words) {
            List<Integer> res = new ArrayList<Integer>();
            if ("".equals(s) || words.length == 0) {
                return res;
            }
            int len = 0;
            HashMap<String, Integer> map = new HashMap<String, Integer>();
            for (int i = 0; i < words.length; i++) {
                map.put(words[i], map.getOrDefault(words[i], 0) + 1);
                len += words[i].length();
            }
            for (int i = 0; i < s.length() - len + 1; i++) {
                String subs = s.substring(i, i + len);
                boolean flag = true;
                HashMap<String, Integer> check = new HashMap<String, Integer>();
                while (flag) {
                    flag = false;
                    for (int j = 0; j < words.length; j++) {
                        int wlen = words[j].length();
                        String key = subs.substring(0, 0 + wlen);
                        if (key.equals(words[j])) {
                            flag = true;
                            check.put(key, check.getOrDefault(key, 0) + 1);
                            if (check.get(key) > map.get(key)) {
                                flag = false;
                                break;
                            }
                            subs = subs.substring(wlen);
                            break;
                        }
                    }
                    if ("".equals(subs) && flag) {
                        res.add(i);
                        break;
                    }
                }
            }

            return res;
        }
        static  class Permutation{
            /**
             * 给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。 按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
             *
             * @param n,k
             */
            public static String getPermutation(int n, int k) {
                StringBuffer sb = new StringBuffer();
                List<Integer> re = new ArrayList<Integer>();
                int[] fib = new int[n + 1];
                fib[0] = 1;
                int fac = 1;
                for (int i = 1; i <= n; i++) {
                    re.add(i);
                    fac *= i;
                    fib[i] = fac;
                }
                k -= 1;
                for (int i = n - 1; i >= 0; i--) {
                    int idx = k / fib[i];
                    sb.append(re.remove(idx));
                    k -= idx * fib[i];
                }
                return sb.toString();

            }

            /**
             * 实现获取下一个排列的函数，算法需要将给定数字序列 重新排列成字典序中下一个更大的排列。 如果不存在下一个更大的排列，
             * 则将数字重新排列成最小的排列（即升序排列）。
             *
             * @param nums
             */
            public static void nextPermutation(int[] nums) {
                if (nums == null || nums.length <= 1) {
                    return;
                }
                int left = -1;
                int right = -1;
                for (int i = nums.length - 1; i >= 0 && left == -1; i--) {
                    for (int j = nums.length - 1; j > i; j--) {
                        if (nums[i] < nums[j]) {
                            left = i;
                            right = j;
                            break;
                        }
                    }
                }
                if (left == -1) {
                    int midIdx = nums.length - 1;
                    for (int i = 0; i < midIdx; i++) {
                        int tmp = nums[i];
                        nums[i] = nums[nums.length - 1 - i];
                        nums[nums.length - 1 - i] = tmp;
                    }
                } else {
                    int tmp = nums[left];
                    nums[left] = nums[right];
                    nums[right] = tmp;
                    Arrays.sort(nums, left + 1, nums.length);
                }
            }
        }


        public String multiply(String num1, String num2) {
            int n1 = num1.length() - 1;
            int n2 = num2.length() - 1;
            if (n1 < 0 || n2 < 0)
                return "";
            int[] mul = new int[n1 + n2 + 2];
            for (int i = n1; i >= 0; i--) {
                for (int j = n2; j >= 0; j--) {
                    int bit = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                    bit += mul[i + j + 1];

                    mul[i + j] = bit / 10;
                    mul[i + j + 1] = bit % 10;
                }
            }
            StringBuffer sb = new StringBuffer();
            int i = 0;
            while (i < mul.length - 1 && mul[i] == 0) {
                i++;
            }
            while (i < mul.length) {
                sb.append(mul[i]);
                i++;
            }
            return sb.toString();
        }

        // 匹配 ？ *
        public static boolean isMatch(String s, String p) {
            int m = s.length();
            int n = p.length();
            boolean[][] flag = new boolean[m + 1][n + 1];
            for (int i = 1; i <= s.length(); i++) {
                flag[0][i] = false;
            }
            for (int i = 1; i <= p.length(); i++) {
                if (p.charAt(i - 1) == '*') {
                    flag[i][0] = flag[i - 1][0];
                    for (int j = 1; j <= s.length(); j++) {
                        flag[i][j] |= flag[i - 1][j];
                        flag[i][j] |= flag[i][j - 1];
                    }
                } else if (p.charAt(i - 1) == '?') {
                    flag[i][0] = false;
                    for (int j = 1; j <= s.length(); j++) {
                        flag[i][j] = flag[i - 1][j - 1];
                    }
                } else {
                    flag[i][0] = false;
                    for (int j = 1; j <= s.length(); j++) {
                        flag[i][j] = s.charAt(j - 1) == p.charAt(j - 1) && flag[i - 1][j - 1];
                    }
                }
            }

            return flag[m][n];
        }

        // 匹配 . *
        public boolean match(char[] str, char[] pattern) {
            int m = str.length, n = pattern.length;
            boolean[][] dp = new boolean[m + 1][n + 1];
            dp[0][0] = true;
            for (int i = 1; i <= n; i++)
                if (pattern[i - 1] == '*')
                    dp[0][i] = dp[0][i - 2];

            for (int i = 1; i <= m; i++)
                for (int j = 1; j <= n; j++)
                    if (str[i - 1] == pattern[j - 1] || pattern[j - 1] == '.')
                        dp[i][j] = dp[i - 1][j - 1];
                    else if (pattern[j - 1] == '*')
                        if (pattern[j - 2] == str[i - 1] || pattern[j - 2] == '.') {
                            dp[i][j] |= dp[i][j - 1]; // a* counts as single a
                            dp[i][j] |= dp[i - 1][j]; // a* counts as multiple a
                            dp[i][j] |= dp[i][j - 2]; // a* counts as empty
                        } else
                            dp[i][j] = dp[i][j - 2]; // a* only counts as empty
            return dp[m][n];
        }

        public static class Parentheses {
            /**
             * 给定一个只包含 '(' 和 ')' 的字符串， 找出最长的包含有效括号的子串的长度。
             *
             * @param s
             * @return
             */
            public static int longestValidParentheses(String s) {
                char[] tmp = s.toCharArray();
                return Math.max(cal(tmp, 0, 1, tmp.length, '('), cal(tmp, -1, -1, -1, ')'));
            }

            private static int cal(char[] tmp, int left, int right, int len, char c) {
                int max = 0, sum = 0, currLen = 0, validLen = 0;
                for (; left != len; left += right) {
                    sum += (tmp[left] == c ? 1 : -1);
                    currLen++;
                    if (sum < 0) {
                        max = max > validLen ? max : validLen;
                        sum = 0;
                        currLen = 0;
                        validLen = 0;
                    } else if (sum == 0) {
                        validLen = currLen;
                    }
                }
                return max > validLen ? max : validLen;
            }

            public static boolean isValidParenthesis(String s) {
                Stack<Character> stack = new Stack<Character>();
                char[] ch = s.toCharArray();
                for (char c : ch) {
                    if (stack.size() == 0) {
                        stack.push(c);
                    } else if (stack.peek() == '(' && c == ')' || stack.peek() == '{' && c == '}') {
                        stack.pop();
                    } else {
                        stack.push(c);
                    }
                }
                return stack.size() == 0;
            }

            public static List<String> generateParenthesis(int n) {
                List<String> result = new ArrayList<String>();
                generateParenthesis(result, "", 0, 0, n);
                return result;
            }

            private static void generateParenthesis(List<String> result, String parenthesis, int i, int j, int n) {
                if (i > n || j > n)
                    return;
                if (i == n || j == n)
                    result.add(parenthesis);
                if (i >= j) {
                    String par1 = new String(parenthesis);
                    generateParenthesis(result, parenthesis + "(", i + 1, j, n);
                    generateParenthesis(result, par1 + ")", i, j + 1, n);
                }
            }
        }

    }

    public static class MathsOperations {
        /**
         * 题意如下，但是数组已经升序排列，所以这里可以考虑二分查找
         * @param nums
         * @return
         */
        public int hIndexII(int[] nums) {
            int l=0, r = nums.length - 1;
            int res=0;

            while(l<=r){
                int mid = l + (r - l) / 2;
                int cnt=nums.length-mid;
                if(cnt<=nums[mid]){
                    res=cnt;
                    r=mid-1;
                }else{
                    l=mid+1;
                }
            }
            return res;
        }
        /**
         * 给定一位研究者论文被引用次数的数组（被引用次数是非负整数）。
         * 编写一个方法，计算出研究者的 h 指数。
         * h 指数的定义：h 代表“高引用次数”（high citations），
         * 一名科研人员的 h 指数是指他（她）的 （N 篇论文中）
         * 总共有 h 篇论文分别被引用了至少 h 次。
         * 其余的 N - h 篇论文每篇被引用次数 不超过 h 次。
         * 例如：某人的 h 指数是 20，这表示他已发表的论文中，
         * 每篇被引用了至少 20 次的论文总共有 20 篇
         * @param citations
         * @return
         */
        public int hIndex(int[] citations) {

            for(int i=0;i<citations.length; i++){
                int h=citations.length-i;
                if(h<=citations[i]){
                    return h;
                }
            }
            return 0;
        }
        /**
         * 编写一个程序，找出第 n 个丑数。
         */
        public int nthUglyNumber(int n) {
            if(n<6){
                return n;
            }
            int pt2=0,pt3=0,pt5=0;
            int[] dp = new int[n];
            dp[0]=1;
            for(int i=1;i<n;i++){
                int next2=dp[pt2]*2,next3=dp[pt3]*3,next5=dp[pt5]*5;
                dp[i]=Math.min(next2,Math.min(next3, next5));
                if(next2==dp[i]){
                    pt2++;
                }
                if(next3==dp[i]){
                    pt3++;
                }
                if(next5==dp[i]){
                    pt5++;
                }
            }
            return dp[n-1];

        }
        /**
         * 编写一个程序判断给定的数是否为丑数。
         * @param num
         * @return
         */
        public boolean isUgly(int nums) {
            if(nums<0) return false;
            while(nums%2==0){
                nums/=2;
            }
            while(nums%3==0){
                nums/=3;
            }
            while(nums%5==0){
                nums/=5;
            }
            return nums==1;
        }

        public int addDigits(int num) {
//            if(num>9){
//                num%=9;
//                if(num==0){
//                    return 9;
//                }
//            }
//            return num;
            return (num-1)%9+1;
        }
        /**
         * 统计一的个数
         * @param n
         * @return
         */
        public int countDigitOne(int n) {
            int num=n,i=1,s=0;
            while(num>0) {
                if(num%10==0){
                    s+=num/10*i;
                }
                if(num%10==1){
                    s+=num/10*i+n%i+1;
                }
                if(num%10>1){
                    s+=Math.ceil(num/10.0)*i;
                }
                num/=10;
                i*=10;
            }
            return s;
        }

        /**
         * 给定一个整数，编写一个函数来判断它是否是 2 的幂次方。
         * @param n
         * @return
         */
        public boolean isPowerOfTwo(int n) {
            if(n<=0){
                return false;
            }
            if((n&(n-1))==0){
                return true;
            }
            return false;
        }
        public boolean isPowerOfTwoR(int n) {
            if(n==1) return true;
            if(n==0) return false;
            if(n%2!=0) return false;
            return isPowerOfTwoR(n/2);
        }
        public static class Caculation{
            public static int calculates(String s) {
                Stack<Integer> stack = new Stack<Integer>();
                char sign='+';
                int num=0;
                int result=0;
                for(int i=0; i<s.length(); i++){
                    char cur=s.charAt(i);
                    if(cur>='0'){
                        num=num*10-'0'+cur;
                    }
                    if((cur<'0'&&cur!=' ')||i==s.length()-1){
                        switch (sign) {
                            case '+':
                                stack.push(num);
                                break;
                            case '-':
                                stack.push(-num);
                                break;
                            case '*':
                                stack.push(stack.pop()*num);
                                break;

                            case '/':
                                stack.push(stack.pop()/num);
                                break;

                        }
                        sign=cur;
                        num=0;
                    }
                }
                while(!stack.isEmpty()){
                    result+=stack.pop();
                }
                return result;
            }
            /**实现一个基本的计算器来计算一个简单的字符串表达式的值。
             * 字符串表达式可以包含左括号 ( ，右括号 )，
             * 加号 + ，减号 -，非负整数和空格 */
            public static int calculate(String s) {
                Stack<Integer> stack = new Stack<Integer>();
                int sign=1,res=0;
                int length=s.length();
                for(int i=0;i<length;i++){
                    char ch=s.charAt(i);
                    if(Character.isDigit(ch)){
                        int cur=ch-'0';
                        while(i+1<length && Character.isDigit(s.charAt(i+1))){
                            cur=cur*10+s.charAt(++i)-'0';
                            //如果下一位还是数字 继续扩大cur
                        }
                        res=res+sign*cur;
                    }else if(ch=='+'){//下一位是加法 标识位为1
                        sign=1;
                    }else if(ch=='-'){//下一位是减法 标识位为-1
                        sign=-1;
                    }else if(ch=='('){
                        stack.push(res);
                        res=0;
                        stack.push(sign);
                        sign=1;
                    }else if(ch==')'){
                        res=stack.pop()*res+stack.pop();
                    }
                }
                return res;
            }
        }

        /**
         * 在二维平面上计算出两个由直线构成的矩形重叠后形成的总面积。
         * @param A
         * @param B
         * @param C
         * @param D
         * @param E
         * @param F
         * @param G
         * @param H
         * @return
         */
        public static int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
            int area=(C-A)*(D-B),area2=(G-E)*(H-F);
            if(C<=E||D<=F||B>=H||A>=G){
                return area+area2;
            }
            int topy=Math.min(D,H),topX=Math.min(C,G);
            int bootomX=Math.max(E,A),bottomY=Math.max(B, F);
            return area+area2-(topX-bootomX)*(topy-bottomY);
        }
        /**
         * 统计所有小于非负整数 n 的质数的数量。
         * @param n
         * @return
         */
        public int countPrimes(int n) {
                boolean[] a= new boolean[n];
                int cnt=0;
                for(int i=2;i<n;i++){
                    if(!a[i]){
                          ++cnt;
                          int k=2;
                        while(i*k<n){
                            a[i*k]=true;
                            k++;
                        }
                    }
                }
                return cnt;
         }
        public boolean isHappy(int n) {
         Map<Integer,Integer> map = new HashMap<Integer,Integer>();
            while (n!=1){
                int tmp=0;
                while(n>0){
                    tmp+=Math.pow(n%10,2);
                    n/=10;
                }
                if(map.containsKey(tmp)){
                    return false;
                }else{
                    map.put(tmp,1);
                }

                 n=tmp;
            }
            return true;
        }
        /**
         * 给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，
         * 返回此范围内所有数字的按位与（包含 m, n 两端点）。
         * @param m,n
         * @return
         */
        public int rangeBitwiseAnd(int m, int n) {
            int offset=0;
            while(m!=n){
                m>>=1;
                n>>=1;
                offset++;
            }
            return n<<offset;
        }
        /**
         * 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数
         * @param n
         * @return
         */
        public int hammingWeight(int n) {
            int ans=0;
            while(n>0){
                ans++;
                n&=n-1;
            }
            return ans;
            //return Integer.bitCount(n);
        }
        /**
         *
         * @param n
         * @return
         */
        public int reverseBits(int n) {
            int a=0;
            for(int i=0;i<=31;i++){
                a+=((1&(n>>i))<<(31-i));
            }
            // int i=32;
            // while(i-->0){
            //     a<<=1;
            //     a+=n&1;
            //     n>>=1;
            // }
            return a;
        }
        /**
         * 根据 逆波兰表示法，求表达式的值。
         * 有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
         * @param tokens
         */
        public int evalRPN(String[] tokens) {
            Stack<Integer> stack=new Stack<>();
            for(String s:tokens){
                if(!s.equals("+")&&!s.equals("-")&&!s.equals("*")&&!s.equals("/")){
                    stack.push(Integer.valueOf(s));
                }else{
                    Integer a=stack.pop();
                    Integer b=stack.pop();
                    if(s.equals("+")){
                        stack.push(b+a);
                    }else if(s.equals("-")){
                        stack.push(b-a);
                    }else if(s.equals("*")){
                        stack.push(a*b);
                    }else{
                        stack.push(b/a);
                    }
                }
            }
            return stack.pop();
        }
        /**
         * @param points
         */
        public int maxPoints(int[][] points) {
            if(points.length<2){
                return points.length;
            }
            Map<Double,Integer> map=new HashMap<>();
            int ans=0;
            for(int i=0;i<points.length-1;i++){
                if(ans>=points.length-1){
                    break;
                }
                int[] point1=points[i];
                int an=0;
                int comma=1;
                for(int j=i+1;j<points.length;i++){
                    if(points[j][0]==point1[0]&&points[j][1]==point1[1]){
                        comma++;
                        continue;
                    }
                    Double slope=getSlope(point1,points[j]);
                    int temp=1;
                    if(map.containsKey(slope)){
                        temp=map.get(slope)+1;
                    }
                    an=Math.max(an, temp);
                    map.put(slope,temp);
                }
                ans=Math.max(ans,an+comma);
                map.clear();
            }
            return ans;
        }
        private Double getSlope(int[] point1,int[] point2){
            //除数为0
            if(point1[1]==point2[1]){
                return null;
            }
            //被除数为0
            if(point1[0]==point2[0]){
                return 0.0;
            }
            return (double)(point1[0]-point2[0])/(point1[1]-point2[1]);
        }

        /**
         * 格雷码
         * @param n
         */
        public static List<Integer> grayCode(int n) {
            List<Integer> result = new ArrayList<Integer>();
            for(int i=0; i<1<<n; i++){
                result.add(i^i>>1);
            }
            return result;
        }
        //实现sqrt()
        public static int mySqrt(int x) {
            //牛顿法
            //折半查找
            if(x==1){
                return 1;
            }
            int min=0;
            int max=x;
            while(max-min>1){
                int mid=(max-min)/2;
                if(x/mid<mid){
                    max=mid;
                }else{
                    min=mid;
                }
            }
            return min;
        }
        //实现 pow(x, n) ，即计算 x 的 n 次幂函数。
        public static double myPow(double x, int n) {
            //return Math.pow(x, n);
            if(n==0) return 1;
            if(n==1) return x;
            boolean isNeg =false;
            if(n<0){
                n=-n;
                isNeg=true;
            }
            double re=myPow(x*x,n/2);
            if(n%2!=0){
                re*=x;
            }
            return isNeg ? 1/re : re;
        }
        public static class Stockeries{
            /**
             * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
             * 设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
             * @param k
             * @param prices
             */
            public int maxProfit(int k, int[] prices) {
                if(k<1) return 0;
                if(k>=prices.length/2) return greedy(prices);
                int[][] dp=new int[k][2];
                for(int i=0;i<k;i++){
                    dp[i][0]=Integer.MIN_VALUE;
                }
                for(int p:prices){
                    dp[0][0]=Math.max(dp[0][0],-p);
                    dp[0][1]=Math.max(dp[0][1],dp[0][0]+p);
                    for(int i=1;i<k;i++){
                        dp[i][0]=Math.max(dp[i-1][1]-p,dp[i][0]);
                        dp[i][1]=Math.max(dp[i][0]+p,dp[i][1]);
                    }
                }
                return dp[k-1][1];
            }
            private int greedy(int[] prices) {
                int max=0;
                for(int i=0;i<prices.length; i++){
                    if(prices[i]>prices[i-1]){
                        max+=prices[i]-prices[i-1];
                    }
                }
                return max;
            }
            /**
             * 给你一个整数数组 nums ，
             * 请你找出数组中乘积最大的连续子数组
             * （该子数组中至少包含一个数字），
             * 并返回该子数组所对应的乘积。
             * @param nums
             */
            public int maxProduct(int[] nums){
                int max = Integer.MIN_VALUE, imax = 1, imin = 1; //一个保存最大的，一个保存最小的。
                for(int i=0; i<nums.length; i++){
                    if(nums[i] < 0){ int tmp = imax; imax = imin; imin = tmp;} //如果数组的数是负数，那么会导致最大的变最小的，最小的变最大的。因此交换两个的值。
                    imax = Math.max(imax*nums[i], nums[i]);
                    imin = Math.min(imin*nums[i], nums[i]);

                    max = Math.max(max, imax);
                }
                return max;
            }
            /**
             * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格
             * 你最多可以完成 两笔 交易。
             * @param prices
             * @return
             */
            public static int maxProfitByTwice(int[] prices) {
                int fBuy=Integer.MIN_VALUE,fSell=0;
                int sBuy=Integer.MIN_VALUE,sSell=0;
                for(int p:prices){
                    fBuy=Math.max(fBuy,0-p);
                    fSell=Math.max(fSell,p+fBuy);
                    sBuy=Math.max(sBuy,fSell-p);
                    sSell=Math.max(sSell,p+sBuy);

                }
                return sSell;
            }
            /**
             * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（
             * @param prices
             * @return
             */
            public static int maxProfitByMulti(int[] prices) {
                if(prices==null){
                    return 0;
                }
                int max=0;
                for(int i=1;i<prices.length;i++){
                    if(prices[i]-prices[i-1]>0){
                        max+=prices[i]-prices[i-1];
                    }
                }
                return max;
            }
            /**
             * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
             * 如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），
             * 设计一个算法来计算你所能获取的最大利润
             */
            public static int maxProfit(int[] prices) {
                if(prices==null){
                    return 0;
                }
                int price=prices[0],max=0;
                for(int p:prices){
                    max=Math.max(max,p-price);
                    price=Math.min(price,p);
                }
                return max;
            }
             //最大子序列和 类似于股票问题
            public static int maxSubArray(int[] nums) {
                int res=nums[0];
                int sum=0;
                for (int i : nums) {
                    if(sum>0){
                        sum+=i;
                    }else{
                        sum=i;
                    }
                    res=Math.max(res,sum);
                }
                return res;
            }
        }
        public double maxProductII(double[] nums) {
            double[] dp=new double[nums.length+1];
            dp[0]=nums[0];
            for(int i=1;i<nums.length;i++){
                // max=Math.max(max,max*nums[i]);
                dp[i]=dp[i-1]*nums[i];
            }

            return Arrays.stream(dp).max().orElse(0);
        }
        public static int climbStairs(int n) {
            //斐波那契数列变种
            int pre=0;
            int pro=1;
            int tmp=0;
            for(int i=0;i<=n;i++){
                tmp=pre+pro;
                pro=pre;
                pre=tmp;
            }
            return tmp;
        }
    }
    public static class DynamicProgramOperations{
        /**
         * 你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。
         * 每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。
         * 你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。
         * 每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。
         * 你的目标是确切地知道 F 的值是多少。
         *
         * 无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？
         *
         * @param K
         * @param N
         * @return
         */
        public int superEggDrop(int K, int N) {
            int[][] dp=new int[K+1][N+1];
            int m=0;
            while(dp[K][m]<N){
                m++;
                for(int k=1;k<=K;k++){
                    dp[k][m]=1+dp[k-1][m-1]+dp[k][m-1];
                }
            }
            return m;
        }
        static class Package{
            //0-1 背包问题的状态转移方程
            //dp(i,j)=max(dp(i-1,j),dp(i-1,j-weight)+value)
            public int knapsack(int W, int N, int[] weights, int[] values){
                if(weights==null||values==null) {
                    return 0;
                }
                int[][] dp=new int[N+1][W+1];
                //W 总重量 N 总数量
                for(int i=1;i<=N;i++){
                    int weight=weights[i-1],value=values[i-1];
                    for(int j=1;j<=W;j++){
                        dp[i][j]=Math.max(dp[i-1][j-1],dp[i-1][j-weight]+value);
                    }
                }
                return dp[N][W];
            }
            //0-1 背包的空间优化
            //dp(j)=max(dp(j),dp(j-weight)+V)
            public int knapsackBySingle(int W, int N, int[] weights, int[] values){
                if(weights==null||values==null) {
                    return 0;
                }
                int[] dp=new int[W+1];
                //W 总重量 N 总数量
                for(int i=1;i<=N;i++){
                    int weight=weights[i-1],value=values[i-1];
                    for(int j=W;j>=1;j--){
                        dp[j]=Math.max(dp[j-1],dp[j-weight]+value);
                    }
                }
                return dp[W];
            }
            //背包问题 不能用贪心解答 因为会造成空间的浪费
            /**
             * - 完全背包：物品数量为无限个
             * - 多重背包：物品数量有限制
             * - 多维费用背包：物品不仅有重量，还有体积，同时考虑这两种限制
             * - 其它：物品之间相互约束或者依赖
             */
            /**
             * leetcode 436
             * 可以看成一个背包大小为 sum/2 的 0-1 背包问题。
             * 给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，
             * 使得两个子集的元素和相等。
             * @param nums
             * @return
             */
            public boolean canPartition(int[] nums){
                int sum=Arrays.stream(nums).sum();
                int W=sum/2;
                boolean[] dp=new boolean[W+1];
                for(int num:nums){
                    for(int j=W;j>=0;j--){
                        dp[j]|=dp[j-num];
                    }
                }
                return dp[W];
            }
        }
        /**
         * 给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。
         * 返回可以使最终数组和为目标数 S 的所有添加符号的方法
         * @param nums
         * @param S
         * @return
         */
        public int findTargetSumWays(int[] nums, int S) {
            if(nums==null){
                return 0;
            }
            return findTargetSumWays(nums,S,0);
        }
        private int findTargetSumWays(int[] nums,int S,int start){
            if(start==nums.length){
                return S==0?1:0;
            }
            return findTargetSumWays(nums, S-nums[start], start+1)+findTargetSumWays(nums,S+nums[start],start+1);
        }
        /**
         *leetcode 494
         */
        public int findTargetSumWaysDp(int[] nums, int S) {
            if(nums==null){
                return 0;
            }

            int sum=Arrays.stream(nums).sum();
            if(sum<S||(sum+S)%2==1){
                return 0;
            }
            int W=(sum+S)/2;
            int[] dp=new int[W+1];
            dp[0]=1;
            for (int num:nums) {
                for(int i=W;i>=num;i--){
                    dp[i]+=dp[i-num];
                }
            }
            return dp[W];
        }

        /**
         * 最长公共子串
         * @param text1
         * @param text2
         * @return
         */
        public int longestCommonSubsequence(String text1, String text2) {
            int m=text1.length(),n=text2.length();
            int[][] dp=new int[m+1][n+1];
            dp[0][0]=0;
            for(int i=1;i<=m;i++){
                for(int j=1;j<=n;j++){
                    if(text1.charAt(i-1)==text2.charAt(j-1)){
                        dp[i][j]=dp[i-1][j-1]+1;
                    }else{
                        //不是同位置 最长公共连续子串 不是最长公共子串
                        dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                    }
                }
            }
            return dp[m][n];
        }
        /**
         * 给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）
         * 使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
         * 任何正整数都可以拆分成不超过4个数的平方和 ---> 答案只可能是1,2,3,4
         * 如果一个数最少可以拆成4个数的平方和，则这个数还满足 n = (4^a)*(8b+7) ---> 因此可以先看这个数是否满足上述公式，如果不满足，答案就是1,2,3了
         * 如果这个数本来就是某个数的平方，那么答案就是1，否则答案就只剩2,3了
         * 如果答案是2，即n=a^2+b^2，那么我们可以枚举a，
         * 来验证，如果验证通过则答案是2
         * 只能是3
         * 因为这里只有四种
         * @param n
         * @return
         */
        public int numSquaresDP(int n){
            int[] dp=new int[n+1];
            dp[0]=0;
            for(int i=1;i<=n;i++){
                int min=1000;
                for(int j=1;j*j<=i;j++){
                    min=Math.min(min,dp[i-j*j]+1);
                }
                dp[i]=min;
            }
            return dp[n];
        }

        //最长递增子序列
        public static int findLongestChain(int[][] pairs) {
            if(pairs.length==0||pairs[0].length==0||pairs==null){
                return 0;
            }
            Arrays.sort(pairs,new Comparator<int[]>(){
                @Override
                public int compare(int[] a,int[] b){
                    if(a[0]==b[0]){
                        return a[1]-b[1];
                    }
                    return a[0]-b[0];
                }
            });
            int[] dp=new int[pairs.length];
            int n=pairs.length;
            Arrays.fill(dp,1);
            for(int i=1;i<n;i++){
                for(int j=0;j<i;j++){
                    if(pairs[i][0]>pairs[j][1]){
                        dp[i]=Math.max(dp[i], dp[j]+1);
                    }
                }
            }
            return Arrays.stream(dp).max().orElse(0);

        }
        /**
         * 最长递增子序列
         * 用二分查找 变为nlogn
         * @param nums
         * @return
         */
        public int lengthOfLIS(int[] nums){
            int n=nums.length;
            int[] dp=new int[n];
            for(int i=0;i<n;i++){
                int max=1;
                for(int j=0;j<i;j++){
                    if(nums[i]>nums[j]){
                        dp[j]=Math.max(max,dp[j]+1);
                    }
                }
            }
            return Arrays.stream(dp).max().orElse(0);
        }
        public  int lengthofLISDP(int[] nums){

            int[] tails=new int[nums.length];
            int len=0;
            for(int n:nums){
                int index=binarySearch(nums,len,n);
                nums[index]=n;
                if(index==nums.length){
                    len++;
                }
            }
            return  len;
        }

        private int binarySearch(int[] nums,int len,int num){
            int l=0,r=len;
            while(l<r){
                int mid=l+(r-l)/2;
                if(nums[mid]==num){
                    return mid;
                }else if(nums[mid]>num){
                    r=mid;
                }else {
                    l=mid+1;
                }
            }
            return l;
        }
        /**
         * 分割整数得最大乘积
         * @param n
         * @return
         */
        public int integerBreak(int n) {
            int[] dp=new int[n+1];
            dp[1]=1;
            for(int i=2;i<n+1;i++){
                for(int j=1;j<i;j++){
                    dp[i]=Math.max(dp[i],Math.max(j*dp[i-j],j*(i-j)));
                }
            }
            return dp[n];
        }
        /**
         * leetcode 376
         * 如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为摆动序列。
         * 第一个差（如果存在的话）可能是正数或负数。少于两个元素的序列也是摆动序列。
         *
         * @param nums
         * @return
         */
        public int wiggleMaxLength(int[] nums) {
            if(nums.length==0||nums==null){
                return 0;
            }
            int up=1,down=1;
            for(int i=1;i<nums.length;i++){
                if(nums[i]>nums[i-1]){
                    up=down+1;
                }else if(nums[i]<nums[i-1]){
                    down=up+1;
                }
            }
            return Math.max(up,down);
        }



        /**
         * leetcode 等差数列划分
         * 如果一个数列至少有三个元素，并且任意两个相邻元素之差相同，则称该数列为等差数列。
         * 从而找到有多少种
         */

        public int numberOfArithmeticSlices(int[] A) {
                if(A==null||A.length==0) return 0;
                int[] dp=new int[A.length+1];
                int total=0;
                for(int i=2;i<A.length; i++){
                    if(A[i-1]-A[i-2]==A[i]-A[i-1]){
                        dp[i]=dp[i-1]+1;
                        total+=dp[i];
                    }
                }
                return total;
                //return Arrays.stream(dp).sum();//这里可以改成遍历 用流的时间较长
        }

        /**
         * leetcode 303 区域和检索 错误点 边界问题
         * @param
         * @return
         */
        class NumArray {
            private int[] sums;
            public NumArray(int[] nums) {
                sums=new int[nums.length+1];
                for(int i=1;i<=nums.length;i++){
                    sums[i]=sums[i-1]+nums[i-1];
                }
            }

            public int sumRange(int i, int j) {
                return sums[j+1]-sums[i];
            }
        }

        /**
         * 入室抢劫问题 间隔抢劫
         * 本质上是斐波那契数列抢劫
         */
        public static class RobProblem{

            /**
             * 环形抢劫
             * @param nums
             * @return
             */
            public int robOfR(int[] nums) {
                if(nums==null||nums.length==0){
                    return 0;
                }
                if(nums.length==1){
                    return nums[0];
                }
                return Math.max(robs(nums,0,nums.length-1),robs(nums,1,nums.length));
            }
            private int robs(int[] nums, int l,int r){
                int pre=0, pro=0;
                for(int i=l;i<r;i++){
                    int cur=Math.max(pre+nums[i],pro);
                    pre=pro;
                    pro=cur;
                }
                return pro;
            }
            /**
             * 这道题算是状态转移方程和动态规划的经典题，写出多种解法
             */
            public int rob(int[] nums) {
                 int pre=0,pro=0;
                 for(int i=0;i<nums.length;i++){
                     int cur=Math.max(pre+nums[i],pro);
                     pre=pro;
                     pro=cur;
                 }
                 return pro;
             }
            public static int robDP(int[] nums){
                int n=nums.length;
                if(n==0){
                    return 0;
                }
                int[] dp=new int[n];
                dp[n-1]=nums[n-1];
                for(int i=n-2;i>=0;i--){
                    dp[i]=Math.max(nums[i]+(i+2>=n?0:dp[i+2]),
                    nums[i+1]+(i+3>=n?0:dp[i+3]));
                }
                return dp[0];
            }

        }
        /**
         * 类似于骑士题
         * @param matrix
         * @return
         */
        public int maximalSquare(char[][] matrix) {
            int m=matrix.length;
            if(m<1) return 0;
            int n=matrix[0].length;
            int max=0;
            int[][] dp = new int[m+1][n+1];
            for(int i=1;i<=m;i++){
                for(int j=1;j<=n; j++){
                    if(matrix[i-1][j-1]=='1'){
                        dp[i][j]=1+Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1]));
                        max=Math.max(max,dp[i][j]);
                    }
                }
            }
            return max*max;
        }
        /**
         * 编写一个函数来计算确保骑士能够拯救到
         * 公主所需的最低初始健康点数。
         * @param dungeon
         * @return
         */
        public int calculateMinimumHP(int[][] dungeon) {
            int row=dungeon.length;
            int col=dungeon[0].length;
            int[][] dp=new int[row][col];
            for(int i=row-1;i>=0;i--){
                for(int j=col-1;j>=0;j--){
                    if(i==row-1&&j==col-1){
                        dp[i][j]=Math.max(1,1-dungeon[i][j]);
                    }
                    if(i==row-1){
                        dp[i][j]=Math.max(1,dp[i][j+1]-dungeon[i][j]);
                    }
                    if(j==col-1){
                        dp[i][j]=Math.max(1,dp[i+1][j]-dungeon[i][j]);
                    }else{
                        dp[i][j]=Math.max(1,Math.min(dp[i][j-1],dp[i-1][j])-dungeon[i][j]);
                    }
                }
            }
            return dp[0][0];
        }

        /**
         * 给定三个字符串 s1, s2, s3,
         * 验证 s3 是否是由 s1 和 s2 交错组成的。
         * @param s1
         * @param s2
         * @param s3
         * @return
         * dp dfs 两种做法 dp优先
         */
        public static boolean isInterleave(String s1, String s2, String s3) {
            int m=s1.length();
            int n=s2.length();
            if(m+n!=s3.length()){
                return false;
            }
            boolean[][] dp=new boolean[m+1][n+1];

            for(int i=0;i<m;i++){
                for(int j=0;j<n;j++){
                    if(i==0&&j==0){
                        dp[i][j]=true;
                        continue;
                    }
                    if(i!=0){
                        dp[i][j]|=dp[i-1][j]&&(s3.charAt(i+j-1)==s1.charAt(i-1));
                    }
                    if(j!=0){
                        dp[i][j]|=dp[i][j-1]&&(s3.charAt(i+j-1)==s2.charAt(j-1));
                    }
                }
            }
            return dp[m][n];
        }

        /**
         * 字符串编辑的动态规划
         */
        static  class StringDP{
            /**
             * 给定两个单词 word1 和 word2，
             * 找到使得 word1 和 word2 相同所需的最小步数，
             * 每步可以删除任意一个字符串中的一个字符。
             * @param word1
             * @param word2
             * @return
             */
            public int minDistanceDele(String word1, String word2) {
                int m=word1.length(),n=word2.length();
                int[][] dp=new int[m+1][n+1];
                for(int i=1;i<m;i++){
                    for(int j=1;j<n;j++){
                        if(word1.charAt(i-1)==word2.charAt(j-2)){
                            dp[i][j]=dp[i-1][j-1]+1;
                        }else{
                            dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                        }
                    }
                }
                return dp[m][n];
            }
            /**
             * 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
             * 你可以对一个单词进行如下三种操作：
             * 插入一个字符
             * 删除一个字符
             * 替换一个字符
             * @param word1
             * @param word2
             * @return
             */
            public static int minDistance(String word1, String word2) {
                int len1=word1.length(),len2=word2.length();
                int[][] dp = new int[len1+1][len2+1];
                for(int i=0;i<=len1;i++){
                    dp[i][0]=i;
                }
                for(int i=0;i<=len1;i++){
                    dp[0][i]=i;
                }
                for(int i=1;i<=len1;i++){
                    for(int j=1;j<=len2;j++){
                        if(word1.charAt(i-1)==word2.charAt(j-1)){
                            dp[i][j]=dp[i-1][j-1];
                        }
                        else{
                            dp[i][j]=1+Math.min(Math.min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]);
                        }
                    }
                }
                return dp[len1][len2];
            }

            public int minSteps(int n){
                if(n==1) return 0;
                for(int i=2;i<=Math.sqrt(n);i++){
                    if(n%i==0) return i+minSteps(n/i);
                }
                return n;
            }
        }

        /**
         * 一个机器人位于一个 m x n 网格的左上角
         * （起始点在下图中标记为“Start” ）。
         * 机器人每次只能向下或者向右移动一步。
         * 机器人试图达到网格的右下角（在下图中标记为“Finish”）。
         * @param m
         * @param n
         * @return
         */
        public static int uniquePaths(int m, int n) {
            int[][] dp = new int[m][n];

            for(int i=0;i<m;i++){
                for(int j=0;j<n;j++){
                    if(i==0||j==0){
                        dp[i][j]=1;
                    }else{
                        dp[i][j]= dp[i-1][j]+dp[i][j-1];
                    }

                }
            }
            return dp[m-1][n-1];
        }

        /**
         * 有障碍的机器人走路方法
         * @param obstacleGrid
         * @return
         */
        public static int uniquePathsWithObstacles(int[][] obstacleGrid) {
            int row=obstacleGrid.length;
            int col=obstacleGrid[0].length;
            if(obstacleGrid[0][0]==1) return 0;
            for(int i=0;i<row;i++){
                for(int j=0;j<col;j++){
                    if(obstacleGrid[i][j]==1){
                        obstacleGrid[i][j]=0;
                        continue;
                    }
                    else{
                        if(i==0&&j==0){
                            obstacleGrid[i][j]=1;
                        }
                        else if(i==0){
                            obstacleGrid[i][j]=obstacleGrid[i][j-1];
                        }else if(j==0){
                            obstacleGrid[i][j]=obstacleGrid[i-1][j];
                        }else{
                            obstacleGrid[i][j]=obstacleGrid[i-1][j]+obstacleGrid[i][j-1];
                        }
                    }

                }
            }
            return obstacleGrid[row-1][col-1];
        }
        /**
         * 动态规划 最小路径和
         * @param grid
         * @return
         */
        public static int minPathSum(int[][] grid) {
            if(grid.length==0||grid[0].length==0){
                return 0;
            }
            int[] dp=new int[grid[0].length];
            for(int i=1;i<grid.length;i++){
                for(int j=0;j<grid[0].length;j++){
                    if(j==0){
                        dp[j]=dp[j];
                    }else if(i==0){
                        dp[j]=dp[j - 1];
                    }else{
                        dp[j]=Math.min(dp[j-1],dp[j]);
                    }
                    dp[j]+=grid[i][j];
                }
            }
            return dp[grid[0].length-1];
        }
        /**
         * 是否是数字
         * @param s
         * @return
         */
        public static boolean isNumber(String s) {
            //状态机
            //取巧的方法
            try {
                s=s.trim();
                if((s.charAt(s.length()-1)<'0'||s.charAt(s.length()-1)>'9')&&s.charAt(s.length()-1)!='.'){
                    return false;
                }
                return true;
            } catch (Exception e) {
                //TODO: handle exception
                return false;
            }
        }
        /**
         * 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
         * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
         * @param digits
         * @return
         */
        public static int[] plusOne(int[] digits) {
            for(int i=digits.length-1;i>=0; i--){
                if(digits[i]!=9){
                    digits[i]++;
                    return digits;
                }
                digits[i]=0;
            }
            int[] tmp = new int[digits.length+1];
            tmp[0] =1;
            return tmp;
        }
        /**
         * 给你两个二进制字符串，返回它们的和（用二进制表示）。
         * 输入为 非空 字符串且只包含数字 1 和 0。
         * @param a
         * @param b
         * @return
         */
        public static String addBinary(String a,String b){
            if(a==null || a.length()==0) return b;
            if(b==null || b.length()==0) return a;

            StringBuffer sb=new StringBuffer();
            int i=a.length()-1;
            int j=b.length()-1;
            int c=0;
            while(i>=0||j>=0){
                if(i>=0) c+=a.charAt(i--)-'0';
                if(j>=0) c+=b.charAt(j--)-'0';
                sb.append(c%2);
                c>>=1;
            }
            String res=sb.reverse().toString();
            return c>0? '1'+res:res;
        }

        /**
         *  给定一个单词数组和一个长度 maxWidth，重新排版单词，
         * 使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
         * @param words
         * @param maxWidth
         * @return
         */
        public static List<String> fullJustify(String[] words, int maxWidth) {
            List<String> res=new ArrayList<String>();
            int idx=0;
            while(idx<words.length) {
                int cur=idx,len=0;
                while(cur<words.length &&len+words[cur].length()+cur-idx<=maxWidth) {
                    len+=words[cur++].length();
                }
                cur--;
                StringBuffer sb=new StringBuffer();
                if(cur==words.length-1){
                    for(int i=idx;i<=cur;i++){
                        sb.append(words[i]);
                        if(i<cur){
                            sb.append(" ");
                        }
                    }
                }else{
                  int base=cur>idx?(maxWidth-len)/(cur-idx) : maxWidth-len;
                  String bStr=generateSpace(base);
                  int left=cur>idx?(maxWidth-len)%(cur-idx) :0;
                  String leftStr=generateSpace(base+1);
                  for(int i=idx;i<=cur;i++){
                      sb.append(words[i]);
                      if(i<cur){
                        sb.append(left>0?leftStr:bStr);
                        left--;
                      }
                  }
                }
                if(sb.length()<maxWidth){
                    sb.append(generateSpace(maxWidth-sb.length()));
                }
                res.add(sb.toString());
                idx=cur+1;
            }
            return res;
        }
        private static String generateSpace(int n){
            char[] cs=new char[n];
            Arrays.fill(cs,' ');
            return new String(cs);
        }
    }
    public static class BSTOperations{
        /**
         * 二叉树的所有路径 回溯 根左右 根为null return 若为叶子结点 根左根右都为null 加入路径
         * @param root
         * @return
         */
        List<String> path = new ArrayList<String>();
        public List<String> binaryTreePaths(TreeNode root) {
            backtrace(root,"");
            return path;
        }
        private void backtrace(TreeNode root,String tmp) {
            if(root==null) return;
            tmp+=root.val;
            if(root.left==null&&root.right==null){
                path.add(tmp);
            }
            if(root.left!=null) backtrace(root.left, tmp+"->");
            if(root.right!=null) backtrace(root.right, tmp+"->");
        }
        /**
         * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
         * @param root
         * @param p
         * @param q
         * @return
         */
        public TreeNode lowestCommonAncestors(TreeNode root, TreeNode p, TreeNode q) {
            if(root==null){
                return null;
            }
            if(root==p||root==q){
                return root;
            }
            TreeNode left=lowestCommonAncestor(root.left,p,q);
            TreeNode right=lowestCommonAncestor(root.right,p,q);
            if(left!=null&&right!=null){
                return root;
            }else if(root.left!=null){
                return left;
            }else if(root.right!=null){
                return right;
            }
            return null;
        }
        /**
         * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
         * @param root
         * @param p
         * @param q
         * @return
         */
        static TreeNode resoflowest;
        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if(root==null){
                return null;
            }
            backtrack(root,p,q);
            return resoflowest;
        }
        private void backtrack(TreeNode root, TreeNode p, TreeNode q){
            if((root.val-p.val)*(root.val-q.val)<=0){
                resoflowest=root;
            }else if(root.val<p.val&&root.val<q.val){
                backtrack(root.right, p, q);
            }else{
                backtrack(root.left, p, q);
            }
        }
        /**
         * 给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。
         * 这里给了k 需要考虑左子树和右子树的内的节点个数，若小于k 则找右子树
         * 若大于k，则找左子树
         * @param root
         * @param k
         * @return
         */
        public int kthSmallest(TreeNode root, int k) {
            int leftSize=getNodeCount(root.left);
            if(leftSize+1==k) return root.val;
            else if(leftSize<k){
                return kthSmallest(root.right,k-leftSize-1);
            }
            return  kthSmallest(root.left,k);
        }
        private int getNodeCount(TreeNode root){
            if(root == null) return 0;
            return getNodeCount(root.left)+getNodeCount(root.right)+1;
        }

        private TreeNode res;
        private int cnt;
        public TreeNode kthNode(TreeNode root,int k){
            inOrder(root,k);
            return res;
        }
        private void inOrder(TreeNode root,int k){
            if(root==null||cnt>=k){
                return;
            }

            inOrder(root.left,k);
            cnt++;
            if(cnt==k){
                res=root;
            }
            inOrder(root.right,k);
        }
        /**
         * 翻转一棵二叉树。
         * @param root
         * @return
        */
        public TreeNode invertTree(TreeNode root) {
            if(root==null){
                return null;
            }
            TreeNode tmp=root.right;
            root.right=invertTree(root.left);
            root.left=invertTree(tmp);
            return root;
        }
        /**
         * 给出一个完全二叉树，求出该树的节点个数。
         * 主要的做法 1 计算深度 2. 递归所有节点
         * @param root
         * @return
         */
        public int countNodes(TreeNode root) {
            if(root==null){
                return 0;
            }
            return countNodes(root.left)+countNodes(root.right)+1;
        }

        public List<Integer> rightSideView(TreeNode root) {
            List<Integer> res=new ArrayList<Integer>();
            if(root==null){
                return res;
            }
            LinkedList<TreeNode> queue=new LinkedList<>();
            queue.add(root);
            while(!queue.isEmpty()){
                int size=queue.size();
                res.add(queue.getFirst().val);
                while(size-->0){
                    TreeNode tmp=queue.pollFirst();
                    if(tmp.right!=null) queue.add(tmp.right);
                    if(tmp.left!=null) queue.add(tmp.left);
                }

            }
            return res;
        }
        public List<Integer> postorderTraversal(TreeNode root) {
            LinkedList<Integer> re=new LinkedList<>();
            if (root == null) {
                return re;
            }
            Stack<TreeNode> queue=new Stack<>();
            queue.add(root);
            while(!queue.isEmpty()){
                TreeNode tmp=queue.pop();
                re.addFirst(tmp.val);
                if(tmp.left!=null){
                    queue.push(tmp.left);
                }
                if(tmp.right!=null){
                    queue.push(tmp.right);
                }

            }
            return re;
        }
        /**
         * 给定一个二叉树，返回它的 前序 遍历。
         * @param root
         * @return
         */
        public List<Integer> preorderTraversal(TreeNode root) {
            List<Integer> re=new ArrayList<>();
            if (root == null) {
                return re;
            }
            Stack<TreeNode> queue=new Stack<>();
            queue.add(root);
            while(!queue.isEmpty()){
                TreeNode tmp=queue.pop();
                re.add(tmp.val);
                if(tmp.right!=null){
                    queue.push(tmp.right);
                }
                if(tmp.left!=null){
                    queue.push(tmp.left);
                }
            }
            return re;
        }
        public LinkRandomNode copyRandomList(LinkRandomNode head) {
            if(head==null) return null;
            LinkRandomNode iter=head;
            Map<LinkRandomNode,LinkRandomNode> map=new HashMap<>();
            while(iter!=null){
                LinkRandomNode randomNode=new LinkRandomNode(iter.val,null,null);
                map.put(iter, randomNode);
                iter=iter.next;
            }
            iter=head;
            while(iter!=null){
                map.get(iter).next=map.get(iter.next);
                map.get(iter).random=map.get(iter.random);
                iter=iter.next;
            }
            return map.get(head);
        }
        /**
         * 给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。
         */
        static int sum;
        public int sumNumbers(TreeNode root) {
            sum=0;
            childSum(0,root);
            return sum;
        }
        private static void childSum(int tmp,TreeNode root){
            if(root==null){
                return;
            }
            int k=root.val+tmp*10;
            if(root.left==null&&root.right==null){
                sum+=k;
            }
            childSum(k, root.left);
            childSum(k, root.right);
        }

        /**
         * 二叉树的最大路径和
         */
        private static int res=Integer.MIN_VALUE;
        public int maxPathSum(TreeNode root) {
            getMax(root);
            return res;
        }
        private static int getMax(TreeNode root){
            if(root==null){
                return 0;
            }
            int left=Math.max(0,getMax(root.left));
            int right=Math.max(0,getMax(root.right));
            res=Math.max(res,root.val+left+right);
            return Math.max(left,right)+root.val;
        }
        /**
         * 给定一个二叉树，原地将它展开为一个单链表。
         */
        public static void flatten(TreeNode root) {
            if(root==null){
                return;
            }
            flatten(root.left);
            flatten(root.right);
            TreeNode tmp=root.right;
            root.right=root.left;
            root.left=null;
            while(root.right!=null)
                root=root.right;
            root.right=tmp;
        }
        /**
         * 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径
         * 同样注意深拷贝和浅拷贝
         */
        static List<List<Integer>> result=new ArrayList<>();
        static List<Integer> list=new ArrayList<>();
        public static List<List<Integer>> pathSum(TreeNode root, int sum) {
            backtrace(root,sum);
            return result;
        }
        private static void backtrace(TreeNode root,int sum){
            if(root==null){
                return ;
            }
            list.add(root.val);
            if(root.left==null&&root.right==null){
                if(sum==root.val){
                    result.add(new ArrayList<>(list));
                    return ;
                }
            }
            backtrace(root.left, sum-root.val);
            backtrace(root.right, sum-root.val);
            list.remove(list.size()-1);
        }
        /**
         * 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
         */
        public static boolean hasPathSum(TreeNode root, int sum) {
            if(root==null){
                return false;
            }
            if(root.left==null&&root.right==null){
                return sum-root.val==0;
            }
            return hasPathSum(root.left, sum-root.val)||hasPathSum(root.right,sum-root.val);
        }
        /**
         * 给定一个二叉树，找出其最小深度
         */
        public static int minDepth(TreeNode root) {
            if(root==null){
                return 0;
            }
            //区分深度 左有 右没有
            if(root.left==null&&root.right!=null){
                return 1+minDepth(root.right);
            }
            if(root.left!=null&&root.right==null){
                return 1+minDepth(root.left);
            }
            return Math.min(minDepth(root.left),minDepth(root.right))+1;
        }

        /**
         * 找出树的最大深度
         * @param root
         * @return
         */
        public static int maxDepth(TreeNode root) {
            if(root == null){
                return 0;
            }
            return Math.max(maxDepth(root.left), maxDepth(root.right))+1;
        }

        /**
         * 给定一个二叉树，判断它是否是高度平衡的二叉树。
         */
        public static boolean isBalanced(TreeNode root) {
            return height(root)>=0;
        }
        private static int height(TreeNode root){
            if(root==null){
                return 0;
            }
            int leftHeight=height(root.left),rightHeight=height(root.right);
            if(leftHeight>=0&&rightHeight>=0&&Math.abs(leftHeight-rightHeight)<=1){
                return Math.max(leftHeight, rightHeight)+1;
            }else{
                return -1;
            }
        }
        /**
         * 给定一个二叉树，返回其节点值自底向上的层次遍历
         */
        public List<List<Integer>> levelOrderBottom(TreeNode root) {
            LinkedList<List<Integer>> result = new LinkedList<List<Integer>>();
            Queue<TreeNode> queue=new LinkedList<>();
            if(root==null){
                return result;
            }
            queue.add(root);
            while(!queue.isEmpty()){
                int cnt=queue.size();
                List<Integer> list=new ArrayList<>();
                while(cnt>0){
                    TreeNode node=queue.poll();
                    list.add(node.val);
                    if(node.left!=null){
                        queue.add(root.left);
                    }
                    if(node.right!=null){
                        queue.add(root.right);
                    }
                    cnt--;
                }
                result.addFirst(list);
            }
            return result;
        }
        /**
         * 给你一个二叉树，请你返回其按 层序遍历 得到的节点值
         */
        public static List<List<Integer>> levelOrder(TreeNode root) {
            List<List<Integer>> result = new ArrayList<List<Integer>>();
            Queue<TreeNode> queue=new LinkedList<>();
            if(root==null){
                return result;
            }
            queue.add(root);
            while(!queue.isEmpty()){
                int cnt=queue.size();
                List<Integer> list=new ArrayList<>();
                while(cnt>0){
                    TreeNode node=queue.poll();
                    list.add(node.val);
                    if(node.left!=null){
                        queue.add(root.left);
                    }
                    if(node.right!=null){
                        queue.add(root.right);
                    }
                    cnt--;
                }

                result.add(list);

            }
            return result;
        }
        /**
         * 之字形遍历二叉树
         * @param root
         * @return
         */
        public static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
            List<List<Integer>> result = new ArrayList<>();
            Deque<TreeNode> queue = new LinkedList<>();
            if(root==null) return result;
            queue.add(root);
            int flag=-1;
            while(!queue.isEmpty()){
                int cnt=queue.size();
                List<Integer> list=new ArrayList<Integer>();
                while(cnt>0){
                    TreeNode node=queue.poll();
                    list.add(node.val);
                    if(node.left!=null){
                        queue.add(node.left);
                    }
                    if(node.right!=null){
                        queue.add(node.right);
                    }
                    cnt--;
                }
                if(flag>0){
                    Collections.reverse(list);
                }
                flag*=-1;
                result.add(list);
            }
            return result;
        }

        /**
         * 给你一个二叉树的根节点 root。设根节点位于二叉树的第 1 层，而根节点的子节点位于第 2 层，依此类推。
         *
         * 请你找出层内元素之和 最大 的那几层（可能只有一层）的层号，并返回其中 最小 的那个。
         *
         * @param root
         * @return
         */
        public int maxLevelSum(TreeNode root) {
            if(root == null) return 0;
            int ans = 0;
            Queue<TreeNode> q = new LinkedList<>();
            q.offer(root);
            int depth = 1;
            int max = Integer.MIN_VALUE;
            while(!q.isEmpty()) {
                int size = q.size();
                int curSum = 0;
                for(int i = 0; i < size; i++) {
                    TreeNode cur = q.poll();
                    curSum += cur.val;
                    if(cur.left != null) {
                        q.offer(cur.left);
                    }
                    if(cur.right != null) {
                        q.offer(cur.right);
                    }
                }
                if(curSum > max) {
                    max = curSum;
                    ans = depth;
                }
                depth++;
            }
            return ans;
        }


        /**
         * 排序链表构建二叉树
         * @param head
         * @return
         * 1. 将链表转数组
         * 2. 快慢指针 找到链表的中点 将其切割 分成左右
         */
        public TreeNode sortedListToBST(ListNode head) {
            if(head==null){
                return null;
            }
            if(head.next==null){
                return new TreeNode(head.val);
            }
            ListNode pre=head;
            ListNode p=pre.next;
            ListNode q=p.next;
            while(q!=null&&q.next!=null){
                pre=pre.next;
                p=p.next;
                q=q.next.next;
            }
            pre.next=null;
            TreeNode root=new TreeNode(pre.val);
            root.left=sortedListToBST(head);
            root.right=sortedListToBST(p.next);
            return root;
        }
        /**
         * 排序数组 构建二叉树
         * @param nums
         * @return
         */
        public static TreeNode sortedArrayToBST(int[] nums) {
            return backtrace(nums,0,nums.length-1);
        }
        private static TreeNode backtrace(int[] nums,int l,int r){
            if(l>r){
                return null;
            }
            int mid=l+(r-l)/2;
            TreeNode root=new TreeNode(nums[mid]);
            root.left=backtrace(nums,l,mid-1);
            root.right=backtrace(nums,mid+1,r);
            return root;
        }
        /**
         * 根据一棵树的前序遍历与中序遍历构造二叉树
         * 面经里面有这道题，用的是回溯+hashmap
         * @param preorder
         * @param inorder
         * @return
         */
        private static HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        public static TreeNode buildTree(int[] preorder, int[] inorder) {
            for (int i = 0; i < inorder.length; i++) {
                map.put(inorder[i], i);
            }
            return backtrace(preorder, 0, preorder.length-1, 0);
        }
        private static TreeNode backtrace(int[] preorder, int l, int r, int inl) {
            if (l > r) {
                return null;
            }
            TreeNode root = new TreeNode(preorder[l]);
            int idx = map.get(preorder[l]);
            int leftSize=idx-inl;
            root.left=backtrace(preorder,l+1,l+leftSize,inl);
            root.right=backtrace(preorder,l+leftSize+1,r,inl+leftSize+1);
            return root;
        }
        /**
         * 后序遍历建立树
         * @param postorder
         * @param inorder
         * @return
         */
        private static HashMap<Integer, Integer> inmap = new HashMap<Integer, Integer>();
        public static TreeNode buildTreePre(int[] inorder, int[] postorder) {
            for(int i=0;i<inorder.length;i++){
                inmap.put(inorder[i], i);
            }
            return backtracePre(postorder, postorder.length-1,0, inorder.length-1);
        }
        private static TreeNode backtracePre(int[] postorder, int r,int inl,int inr){
            if(inl>inr){
                return null;
            }
            TreeNode root=new TreeNode(postorder[r]);
            int ridx=inmap.get(root.val);
            int rightSize=inr-ridx;
            root.left=backtrace(postorder,r-rightSize-1,inl,ridx-1);
            root.right=backtrace(postorder,r-1,ridx+1,inr);
            return root;
        }
        /**
         * 给定一个二叉树，检查它是否是镜像对称的
         */

        public static boolean isSymmetric(TreeNode root) {
            if(root==null){
                return true;
            }
            return isSymmetric(root.left,root.right);
        }
        private static boolean isSymmetric(TreeNode left,TreeNode right){
            if(left==null&&right==null){
                return true;
            }
            if(left==null||right==null){
                return false;
            }
            if(left.val!=right.val){
                return false;
            }
            return isSymmetric(left.left,right.right)&&isSymmetric(left.right,right.left);
        }
        /**
         * 给定两个二叉树，编写一个函数来检验它们是否相同。
         */
        public static boolean isSameTree(TreeNode p, TreeNode q) {
            if(q==null&&q==null){
                return true;
            }
            if(q==null) return false;
            if(p==null) return false;
            if(p.val==q.val){
                return isSameTree(p.left, q.left)&&isSameTree(p.right, q.right);
            }else{
                return false;
            }

        }
        /**
         * 恢复二叉树
         */
        static TreeNode t1,t2,pre;
        public static void recoverTree(TreeNode root) {
            inorder(root);
            int tmp=t1.val;
            t1.val=t2.val;
            t2.val=tmp;
        }
        private static void inorder(TreeNode root){
            if(root==null) return;
            inorder(root.left);
            if(pre!=null&&pre.val>root.val){
                if(t1==null) t1=pre;
                t2=root;
            }
            pre=root;
            inorder(root.right);
        }
        static double last=-Double.MAX_VALUE;
        public static boolean isValidBST(TreeNode root) {
            if(root==null){
                return true;
            }
            if(isValidBST(root.left)){
                if(root.val>last){
                    last=root.val;
                    return isValidBST(root.right);
                }
            }
            return false;
        }
        public List<Integer> inorderTraversal(TreeNode root) {
            List<Integer> result=new ArrayList<>();
            Stack<TreeNode> stack=new Stack<>();
            while(root!=null||!stack.isEmpty()){
              if(root!=null){
                  stack.push(root);
                  root=root.left;
              }else{
                  root=stack.pop();
                  result.add(root.val);
                  root=root.right;
              }
            }
            return result;
        }

        /**
         *有多少子数
         * @param n
         * @return
         */
        public static int numTrees(int n) {
            if(n==0){
                return 0;
            }
            List<TreeNode> res=backtrace(1,n);
            return res.size();
        }
        /**
         * 生成树
         * @param n
         * @return
         */
        public static List<TreeNode> generateTrees(int n) {
            if(n==0){
                return new ArrayList<>();
            }
            return backtrace(1,n);
        }
        private static List<TreeNode> backtrace(int left,int right){
            List<TreeNode> res=new ArrayList<>();
            if(left>right){
                res.add(null);
                return res;
            }
            for(int i=left;i<=right;i++){
                List<TreeNode> subLeftNodes=backtrace(left, i-1);
                List<TreeNode> subRightNodes=backtrace(i+1, right);
                for(TreeNode l:subLeftNodes){
                    for(TreeNode r:subRightNodes){
                        TreeNode root=new TreeNode(i);
                        root.left=l;
                        root.right=r;
                        res.add(root);
                    }
                }
            }
            return res;
        }
        /**
         * 生成二叉搜索树
         */
        public static List<TreeNode> generateBST(int n){
            if(n==0){
                return new ArrayList<>();
            }
            return generateBST(1,n);
        }
        private static List<TreeNode> generateBST(int left,int right){
            List<TreeNode> res=new ArrayList<>();
            if(left>right){
                res.add(null);
                return res;
            }
            for(int i=left;i<right;i++){
                List<TreeNode> leftTree=generateBST(left,i-1);
                List<TreeNode> rightTree=generateBST(i+1,right);
                for(TreeNode l:leftTree){
                    for(TreeNode r:rightTree){
                        if(i>l.val&&i<r.val){
                            TreeNode root=new TreeNode(i);
                            root.left=l;
                            root.right=r;
                            res.add(root);
                        }
                    }
                }
            }
            return res;
        }


        //116
        /**
         * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。
         * 如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
         * 初始状态下，所有 next 指针都被设置为 NULL。
         */
        public static TreeLinkNode connect(TreeLinkNode root){
            if(root==null||root.left==null){
                return root;
            }
            root.left.next=root.right;
            if(root.next!=null){
                root.right.next=root.next.left;
            }
            connect(root.left);
            connect(root.right);
            return root;
        }
        /**
         * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。
         * 如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
         * 初始状态下，所有 next 指针都被设置为 NULL。
         * 通用解法
         */
        public static TreeLinkNode connects(TreeLinkNode root){
            if(root==null){
                return null;
            }
            LinkedList<TreeLinkNode> queue=new LinkedList<>();
            queue.add(root);
            while(!queue.isEmpty()){
                int size=queue.size();
                while(size-->0){
                    TreeLinkNode node=queue.remove();
                    if(size>0){
                        node.next=queue.peek();
                    }
                    if(node.left!=null){
                        queue.add(node.left);
                    }
                    if(node.right!=null){
                        queue.add(node.right);
                    }
                }
            }
            return root;
        }

    }
    public static class BaseAlgorithm{
        static  class QuickSort{
            public  static  void quickSort(int[] nums){

            }
        }

        /**
         * 二分搜索
         */
        static class BinarySearch{
            public  static  int binarySearch(int[] nums,int target){


                int l=0,r=nums.length-1;
                while(l<r){
                    int mid=(r-l)/2+l;
                    if(nums[mid]==target){
                        return nums[mid];
                    }
                    if(nums[mid]>target){
                        r=mid;
                    }else{
                        l=mid+1;
                    }
                }
                return nums[l];
            }

        }

        /**
         * 归并排序
         */
        static class MergeSort{
            public static void mergeSort(int[] nums){
                int[] left= new int[nums.length/2];
                System.arraycopy(nums,0,left,0,nums.length/2);
                mergeSort(left);
                int[] right= new int[nums.length-nums.length/2];
                System.arraycopy(nums,nums.length/2,right,0,nums.length-nums.length/2);
                mergeSort(right);
                int[] tmp=merge(left,right);
                System.arraycopy(tmp,0,nums,nums.length-1,nums.length);
            }
            private static int[] merge(int[] left, int[] right){
                int[] res=new int[left.length+right.length];
                int curLeft=0;
                int curRight=0;
                int curRes=0;
                while(curLeft<left.length&&curRight<right.length){
                    if(left[curLeft]<right[curRight]){
                        res[curRes++]=left[curLeft++];
                    }else{
                        res[curRes++]=right[curRight++];
                    }
                }
                while(curLeft<left.length){
                    res[curRes++]=left[curLeft++];
                }
                while(curRight<right.length){
                    res[curRes++]=right[curRight++];
                }
                return res;
            }
        }
    }

    public static class SystemDesign{
        public static class Stacks{
            public  interface  MyStack<T> extends Iterable<T>{
                void push(T t);
                T pop();
                boolean isEmpty();
                void empty();
            }

            public  class ArrayStack<T> implements MyStack<T> {
                final  static int Cap=10;
                T[] item= (T[]) new Object[Cap];
                int size=0;
                @Override
                public void push(T t) {

                }

                @Override
                public T pop() {
                    return null;
                }

                @Override
                public boolean isEmpty() {
                    return false;
                }

                @Override
                public void empty() {

                }

                @Override
                public Iterator<T> iterator() {
                    return null;
                }
            }
            public class LinkedStack<T> implements  MyStack<T>{
                ListNode node=new  ListNode(-1);
                @Override
                public void push(T t) {
                    node.next=new ListNode((Integer) t);

                }

                @Override
                public T pop() {
                    return null;
                }

                @Override
                public boolean isEmpty() {
                    return false;
                }

                @Override
                public void empty() {

                }

                @Override
                public Iterator<T> iterator() {
                    return null;
                }
            }
        }
        public  static  class ArrayLists {
            public interface MyLists<T> extends  Cloneable{
                void add(T t);
                T remove();
                T remove(int index);

            }
            public  class  ArrayList<T> implements  MyLists<T>{

                @Override
                public void add(T t) {

                }

                @Override
                public T remove() {
                    return null;
                }

                @Override
                public T remove(int index) {
                    return null;
                }
            }
        }
        public  static  class Queues{
        }


        /**
        * Your WordDictionary object will be instantiated and called as such:
        * WordDictionary obj = new WordDictionary();
        * obj.addWord(word);
        * boolean param_2 = obj.search(word);
        */
        class PeekingIterator implements Iterator<Integer> {
            Iterator<Integer> iter;
            Integer cash=null;
            public PeekingIterator(Iterator<Integer> iterator) {
                // initialize any member here.
                iter=iterator;
            }

            // Returns the next element in the iteration without advancing the iterator.
            public Integer peek() {
                if(cash==null){
                    cash=iter.next();
                }
                return cash;
            }

            // hasNext() and next() should behave the same as in the Iterator interface.
            // Override them if needed.
            @Override
            public Integer next() {
                if(cash==null){
                    cash=iter.next();
                }
                Integer tmp=cash;
                cash=null;
                return tmp;
            }

            @Override
            public boolean hasNext() {
                return cash!=null||iter.hasNext();
            }
    }

        /**
         * 栈实现队列
         */
        class MyQueue {
            Stack<Integer> stackIn;
            Stack<Integer> stackOut;
            /** Initialize your data structure here. */
            public MyQueue() {
                stackIn = new Stack<>();
                stackOut = new Stack<>();
            }

            /** Push element x to the back of queue. */
            public void push(int x) {
                stackIn.push(x);

            }

            /** Removes the element from in front of queue and returns that element. */
            public int pop() {
                if(stackOut.isEmpty()){
                    while(!stackIn.isEmpty()){
                        stackOut.push(stackIn.pop());
                    }
                }
                return stackOut.pop();
            }

            /** Get the front element. */
            public int peek() {
                if(stackOut.isEmpty()){
                    while(!stackIn.isEmpty()){
                        stackOut.push(stackIn.pop());
                    }
                }
                return stackOut.peek();
            }

            /** Returns whether the queue is empty. */
            public boolean empty() {
                return stackIn.isEmpty()&&stackOut.isEmpty();
            }
        }

        /**
         * 队列实现栈
         * @param <T>
         */
        class MyStack<T> {
            Queue<Integer> pushqueue;
            Queue<Integer> popqueue;
            /** Initialize your data structure here. */
            public MyStack() {
                pushqueue=new LinkedList<>();
                popqueue=new LinkedList<>();
            }

            /** Push element x onto stack. */
            public void push(int x) {
                pushqueue.add(x);
                while(!popqueue.isEmpty()){
                    pushqueue.add(popqueue.poll());
                }
                Queue<Integer> tmp=pushqueue;
                pushqueue=popqueue;
                popqueue=tmp;
            }

            /** Removes the element on top of the stack and returns that element. */
            public int pop() {
               return popqueue.poll();
            }

            /** Get the top element. */
            public int top() {
              return  popqueue.peek();
            }

            /** Returns whether the stack is empty. */
            public boolean empty() {
                return popqueue.isEmpty();
            }
        }

        /**
         * LRU Cache
         */
        class LRUCache {
            int cap;
            LinkedHashMap<Integer, Integer> cache;
            public LRUCache(int capacity) {
                this.cap=capacity;
                //为什么是0.75
                //提高空间利用率和 减少查询成本的折中，主要是泊松分布，0.75的话碰撞最小，
                cache=new LinkedHashMap<Integer, Integer>(cap,0.75f,true){
                    private static final long serialVersionUID = 7438887847067727547L;
                    @Override
                    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
                        return cache.size()>cap;
                    }
                };
            }

            public int get(int key) {
                return cache.getOrDefault(key,-1);
            }

            public void put(int key, int value) {
                cache.put(key,value);
            }
        }

        /**
         * 最小栈
         */
        class MinStack {
            private Node head;
            /** initialize your data structure here. */
            public MinStack() {

            }
            private class Node{
                int val;
                int min;
                Node next;
                Node(int value,int minvalue){
                    this(value,minvalue,null);
                }
                Node(int value,int minvalue,Node next){
                    this.val = value;
                    this.min = minvalue;
                    this.next = next;
                }
            }
            public void push(int x) {
                if(head== null){
                    head=new Node(x,x);
                }else{
                    head=new Node(x,Math.min(x,head.min),head);
                }
            }

            public void pop() {
                head=head.next;
            }

            public int top() {
                return head.val;
            }

            public int getMin() {
                return head.min;
            }
        }

        /**
         * 二叉搜索树
         */
        class BSTIterator {
            private TreeNode node;
            private Queue<Integer> queue;
            public BSTIterator(TreeNode root) {
                this.node = root;
                this.queue = new LinkedList<Integer>();
                midOrder(node);
            }

            /** @return the next smallest number */
            public int next() {
                return queue.poll();
            }

            /** @return whether we have a next smallest number */
            public boolean hasNext() {
                return !queue.isEmpty();
            }
            private void midOrder(TreeNode root){
                if(root==null){
                    return;
                }

                midOrder(root.left);
                queue.add(root.val);
                midOrder(root.right);
            }
        }

        /**
         * 字典树
         */
        class Trie {
            public class TireNode{
                private boolean isUsed;
                private TireNode[] next;
                TireNode() {
                    isUsed = false;
                    next=new TireNode[26];
                }
            }
            private TireNode root;
            /** Initialize your data structure here. */
            public Trie() {
                root=new TireNode();
            }

            /** Inserts a word into the trie. */
            public void insert(String word) {
                TireNode cur=root;
                for(int i=0,len=word.length(),ch;i<len;i++) {
                    ch=word.charAt(i)-'a';
                    if(cur.next[ch]==null)
                        cur.next[ch]=new TireNode();
                    cur=cur.next[ch];
                }
                cur.isUsed=true;
            }

            /** Returns if the word is in the trie. */
            public boolean search(String word) {
                TireNode cur=root;
                for(int i=0,len=word.length(),ch;i<len;i++) {
                    ch=word.charAt(i)-'a';
                    if(cur.next[ch]==null){
                        return false;
                    }
                    cur=cur.next[ch];
                }
                return cur.isUsed;
            }

            /** Returns if there is any word in the trie that starts with the given prefix. */
            public boolean startsWith(String prefix) {
                TireNode cur=root;
                for(int i=0,len=prefix.length(),ch;i<len;i++) {
                    ch=prefix.charAt(i)-'a';
                    if(cur.next[ch]==null){
                        return false;
                    }
                    cur=cur.next[ch];
                }
                return true;
            }
        }



    }
}