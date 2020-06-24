import java.util.*;
public class AlgoII {
    public static class ListNode{
        int val;
        ListNode next;
        ListNode(int x){
            val = x;
        }
    }
    public static class TreeNode{
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x){
            val = x;
        }
    }
    public static void main(String[] args) {
        int[] data ={2, 3, 1, 0, 2, 5};

        int[] result=new int[1];
        System.out.println(duplicate(data,data.length,result));

        int[][] matrix={{1, 4, 7, 11, 15}, {2, 5, 8, 12, 19}, {3, 6, 9, 16, 22}, {10, 13, 14, 17, 24}, {18, 21, 23, 26, 30}};
        int target=5;
        System.out.println(isInMatrix(matrix, target));

        StringBuffer sb=new StringBuffer();
        sb.append("A B");
        System.out.println(replaceSpace(sb));
    }
    //数组中重复的数字
    public static boolean duplicate(int[] nums,int length,int[] dup) {
    //    return false;
        if(nums==null ||length<=0){
            return false;
        }
        for(int i=0;i<length;i++){
            while(nums[i]!=i){
                dup[0]=nums[i];
                return true;
            }
            swap(nums,i,nums[i]);
        }
        return false;
    }
    private static void swap(int[] nums,int i,int j){
        int tmp=nums[i];
        nums[i]=nums[j];
        nums[j]=tmp;
    }
    //二维数组中的查找


    public static boolean isInMatrix(int[][] matrix,int target){
        if(matrix.length==0||matrix[0].length==0){
            return false;
        }
        int cols=matrix.length,rows=matrix[0].length;
        int col=0,row=rows-1;
        while(col<cols&&row>=0){
            if(target==matrix[row][col]){
                return true;
            }else if(target<matrix[row][col]){
                row--;
            }else{
                col++;
            }
        }
        return false;
    }
    // 替换空格
    public static String replaceSpace(StringBuffer str) { 
        int P1=str.length()-1;
        for(int i=0;i<P1;i++){
            if(str.charAt(i)==' '){
                str.append("  ");
            }
        }
        int P2=str.length()-1;
        while(P1>=0&&P2>P1){
            char c=str.charAt(P1--);
            if(c==' '){
                str.setCharAt(P2--,'0');
                str.setCharAt(P2--,'2');
                str.setCharAt(P2--,'%');
            }else{
                str.setCharAt(P2--,c);
            }
        }
        return str.toString();
    }
    //从尾到头打印链表
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) { 
        ArrayList<Integer> list = new ArrayList<Integer>();
        if(listNode == null){
            return list;
        }
        Stack<Integer> stack = new Stack<Integer>();
        while(listNode != null) {
            stack.push(listNode.val);
            listNode=listNode.next;
        }
        
        while(stack.isEmpty()){
            list.add(stack.pop());
        }
        return list;
    }
    //重建二叉树
    // 先说思路
    // 回溯法  前序编理 根左右
    private static HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) { 
        //return null;
        for(int i=0;i<in.length;i++){
            map.put(in[i],i);
        }
        return  backtrace(pre,0,pre.length-1,0);
    }
    private static TreeNode backtrace(int[] pre,int l,int r,int inl){
        if(l>r){
            return null;
        }
        TreeNode root=new TreeNode(pre[l]);
        int idx=map.get(pre[l]);
        int leftSize=idx-l;
        root.left=backtrace(pre,l+1,l+leftSize,inl);
        root.right=backtrace(pre,l+leftSize+1,r,inl+leftSize+1);
        return root;
    }

}