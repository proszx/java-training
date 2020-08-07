package com.company.algorithm;



import java.util.*;
public class AlgoII {
    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public static void main(String[] args) {
        int[] data = {3, 4, 5, 6, 1, 2};

        int[] result = new int[1];
        System.out.println(duplicate(data, data.length, result));

        int[][] matrix = {{1, 4, 7, 11, 15}, {2, 5, 8, 12, 19}, {3, 6, 9, 16, 22}, {10, 13, 14, 17, 24}, {18, 21, 23, 26, 30}};
        int target = 5;
        System.out.println(isInMatrix(matrix, target));

        StringBuffer sb = new StringBuffer();
        sb.append("A B");
        System.out.println(replaceSpace(sb));

        System.out.println(minNumberInRotateArray(data));
        System.out.println(integerBreak(6));

        System.out.println(Power(2.2, 3));

    }

    public static int integerBreak(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i - 1; j++) {
                dp[i] = Math.max(dp[i], Math.max(j * dp[i - j], j * (i - j)));
            }
        }
        return dp[n];
    }

    //数组中重复的数字
    public static boolean duplicate(int[] nums, int length, int[] dup) {
        //    return false;
        if (nums == null || length <= 0) {
            return false;
        }
        for (int i = 0; i < length; i++) {
            while (nums[i] != i) {
                dup[0] = nums[i];
                return true;
            }
            swap(nums, i, nums[i]);
        }
        return false;
    }

    private static void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
    //二维数组中的查找


    public static boolean isInMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int cols = matrix.length, rows = matrix[0].length;
        int col = 0, row = rows - 1;
        while (col < cols && row >= 0) {
            if (target == matrix[row][col]) {
                return true;
            } else if (target < matrix[row][col]) {
                row--;
            } else {
                col++;
            }
        }
        return false;
    }

    // 替换空格
    public static String replaceSpace(StringBuffer str) {
        int P1 = str.length() - 1;
        for (int i = 0; i < P1; i++) {
            if (str.charAt(i) == ' ') {
                str.append("  ");
            }
        }
        int P2 = str.length() - 1;
        while (P1 >= 0 && P2 > P1) {
            char c = str.charAt(P1--);
            if (c == ' ') {
                str.setCharAt(P2--, '0');
                str.setCharAt(P2--, '2');
                str.setCharAt(P2--, '%');
            } else {
                str.setCharAt(P2--, c);
            }
        }
        return str.toString();
    }

    //从尾到头打印链表
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> list = new ArrayList<Integer>();
        if (listNode == null) {
            return list;
        }
        Stack<Integer> stack = new Stack<Integer>();
        while (listNode != null) {
            stack.push(listNode.val);
            listNode = listNode.next;
        }

        while (stack.isEmpty()) {
            list.add(stack.pop());
        }
        return list;
    }

    //重建二叉树
    // 先说思路
    // 回溯法  前序编理 根左右
    private static HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();

    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        //return null;
        for (int i = 0; i < in.length; i++) {
            map.put(in[i], i);
        }
        return backtrace(pre, 0, pre.length - 1, 0);
    }

    private static TreeNode backtrace(int[] pre, int l, int r, int inl) {
        if (l > r) {
            return null;
        }
        TreeNode root = new TreeNode(pre[l]);
        int idx = map.get(pre[l]);
        int leftSize = idx - l;
        root.left = backtrace(pre, l + 1, l + leftSize, inl);
        root.right = backtrace(pre, l + leftSize + 1, r, inl + leftSize + 1);
        return root;
    }
    //旋转数组中最小的数字

    public static int minNumberInRotateArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] <= nums[r]) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return nums[l];
    }

    //是否有路径
    private static int[][] next = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    static int rows;
    static int cols;

    public boolean hashPath(char[] array, int rows, int cols, char[] str) {
        this.cols = cols;
        this.rows = rows;
        boolean[][] flag = new boolean[rows][cols];
        char[][] matrix = buildMatrix(array, rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (backtrace(matrix, str, i, j, 0, flag)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean backtrace(char[][] matrix, char[] str, int i, int j, int pathLen, boolean[][] flag) {
        if (pathLen == str.length) {
            return true;
        }
        if (i < 0 || j < 0 || i >= rows || j >= cols || flag[i][j] || matrix[i][j] != str[pathLen]) {
            return false;
        }
        flag[i][j] = true;
        for (int[] n : next) {
            if (backtrace(matrix, str, i + n[0], j + n[1], pathLen + 1, flag)) {
                return true;
            }
        }
        flag[i][j] = false;
        return false;
    }

    private char[][] buildMatrix(char[] array, int rows, int cols) {
        char[][] matrix = new char[rows][cols];
        for (int i = 0, idx = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = array[idx++];
            }
        }
        return matrix;
    }
    //机器人的运动范围 能够到达多少个格子


    public static class MovingCount {
        private static int[][] next = {{1, 0}, {-1, 0}, {0, -1}, {0, 1}};
        private static int threshold;
        private static int cnt;
        private static int cols;
        private static int rows;

        public static int movingCount(int thres, int row, int col) {
            cols = col;
            rows = row;
            threshold = thres;
            cnt = 0;
            int[][] matrix = buildMatrix(threshold);
            boolean[][] flag = new boolean[row][col];
            dfs(matrix, flag, 0, 0);
            return cnt;

        }

        private static void dfs(int[][] matrix, boolean[][] flag, int r, int c) {
            if (r <= 0 || c <= 0 || r < rows || c >= cols || !flag[r][c]) {
                return;
            }
            flag[r][c] = true;
            if (matrix[r][c] > threshold) {
                return;
            }
            cnt++;
            for (int[] n : next) {
                dfs(matrix, flag, r + n[0], c + n[1]);
            }
        }

        private static int[][] buildMatrix(int threshold) {
            int[] tmp = new int[Math.max(rows, cols)];
            for (int i = 0; i < tmp.length; i++) {
                int n = i;
                while (n >= 0) {
                    tmp[i] = n % 10;
                    n /= 10;
                }
            }
            int[][] matrix = new int[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i][j] = tmp[i] + tmp[j];
                }
            }
            return matrix;
        }

    }

    //二进制中1的个数
    public static int NumberOf1(int n) {
        int cnt = 0;
        while (n != 0) {
            cnt++;
            n &= (n - 1);
        }
        //return cnt;
        return Integer.bitCount(n);
    }

    //给定一个 double 类型的浮点数 base 和 int 类型的整数 exponent，求 base 的 exponent 次方
    public static double Power(double base, int exponent) {
        if (exponent == 0) {
            return 1;
        }
        if (exponent == 1) {
            return base;
        }
        boolean isNeg = false;
        if (base < 0) {
            isNeg = true;
            base = -base;
        }
        double result = Power(base * base, exponent / 2);
        if (exponent % 2 == 1) {
            result *= base;
        }
        return isNeg ? result * (-1) : result;

    }

    /**
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。
     */
    public boolean verifyPostorder(int[] postorder) {
        if (postorder.length < 2) {
            return true;
        }
        return bracktrace(0, postorder.length - 1, postorder);
    }

    private boolean bracktrace(int l, int r, int[] postorder) {
        if (l >= r) return true;
        int rootValue = postorder[r];
        int curIndex = l;
        while (curIndex < r && postorder[curIndex] < rootValue) {
            curIndex++;
            //左子树索引
        }
        for (int i = curIndex; i < r; i++) {
            if (postorder[i] < rootValue) {
                return false;
            }
        }
        return bracktrace(l, curIndex - 1, postorder) && bracktrace(curIndex, r - 1, postorder);

    }

    /**
     * 打印从 1 到大的 n 位数
     * 回溯法
     * 1.满足返回条件 开始打印
     * 2. 往素组里面装数字
     *
     * @param n
     */
    public void print1ToMaxOfNDigits(int n) {
        if (n <= 0) {
            return;
        }
        char[] tmp = new char[n];
        print1TomaxOfDigits(0, tmp);
    }

    private void print1TomaxOfDigits(int digit, char[] tmp) {
        if (digit == tmp.length) {
            printNumber(tmp);
            return;
        }
        for (int i = digit; i < tmp.length; i++) {
            tmp[i] = (char) (i + '0');
            print1TomaxOfDigits(digit + 1, tmp);
        }
    }

    private void printNumber(char[] tmp) {
        int idx = 0;
        while (idx < tmp.length && tmp[idx] == '0') {
            idx++;
        }
        while (idx < tmp.length) {
            System.out.print(tmp[idx++]);
        }
        System.out.println();
    }

    public ListNode deleteNode(ListNode head, ListNode tobeDelete) {
        if (head == null || tobeDelete == null) {
            return null;
        }
        if (tobeDelete.next != null) {
            ListNode next = tobeDelete;
            tobeDelete.val = next.val;
            tobeDelete.next = next.next;
        } else {
            if (head == tobeDelete) {
                return null;
            }
            ListNode cur = head;
            while (cur != tobeDelete) {
                cur = cur.next;
            }
            cur.next = null;
        }
        return head;
    }

    public ListNode deleteDuplication(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode next = pHead.next;
        if (pHead.val == next.val) {
            while (pHead != null && pHead.val == next.val) {
                next = next.next;

            }
            return deleteDuplication(next);
        } else {
            pHead.next = deleteDuplication(pHead.next);
            return pHead;
        }
    }

    /**
     * 请实现一个函数用来匹配包括 '.' 和 '*' 的正则表达式。模式中的字符 '.'
     * 表示任意一个字符，而 '*' 表示它前 面的字符可以出现任意次（包含 0
     *
     * @param str
     * @param pattern
     * @return
     */
    public boolean isMatch(String str, String pattern) {
        int m = str.length(), n = pattern.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        for (int i = 1; i < n + 1; i++) {
            if (pattern.charAt(i - 1) == '*') {
                dp[0][i] = dp[0][i - 2];
            }
        }
        for (int i = 1; i < m + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                if (str.charAt(i - 1) == pattern.charAt(j - 1) || pattern.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                if (pattern.charAt(j - 1) == '*') {
                    if (str.charAt(i - 1) == pattern.charAt(j - 2) || pattern.charAt(j - 2) == '.') {
                        dp[i][j] |= dp[i][j - 1];//单个相等
                        dp[i][j] |= dp[i - 1][j];//复合相等 和前一位也相同
                        dp[i][j] |= dp[i][j - 2];//a*为空
                    }
                } else {
                    dp[i][j] |= dp[i][j - 2];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 匹配 ？ *
     * '?' 可以匹配任何单个字符。
     * '*' 可以匹配任意字符串（包括空字符串）。
     *
     * @param str
     * @param pattern
     * @return
     */
    public boolean isMatch(String str, char[] pattern) {
        int m = str.length(), n = pattern.length;
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i < n + 1; i++) {
            if (pattern[i - 1] == '*') {
                dp[0][i] = dp[0][i - 1];
            }
        }
        for (int i = 1; i < m + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                if (pattern[j - 1] == '*') {
                    dp[i][j] |= dp[i - 1][j];
                    dp[i][j] |= dp[i][j - 1];
                } else if (pattern[j - 1] == '?') {
                    dp[i][j] |= dp[i - 1][j - 1];
                } else if (str.charAt(i - 1) == pattern[j - 1]) {
                    dp[i][j] |= dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 表示数值的字符串
     *
     * @param str
     * @return
     */
    public boolean isNumeric(char[] str) {
        if (str == null) {
            return false;
        }
        return str.toString().matches("[+-]?\\d*(\\.\\d+)?([eE][+-]?\\d+)?");
    }

    /**
     * 使奇数位于偶数前面
     *
     * @param nums
     */
    public void reOrderArray(int[] nums) {
        int[] tmp = nums;
        int cnt = 0;
        for (int n : nums) {
            if (n % 2 == 0) {
                cnt++;
            }
        }
        int i = 0;
        for (int n : tmp) {
            if (n % 2 == 1) {
                nums[i++] = n;
            } else {
                nums[cnt++] = n;
            }
        }
    }

    /**
     * 链表的倒数第k个节点
     */
    public ListNode FindKthToTail(ListNode head, int k) {
        if (head == null || k < 0) {
            return null;
        }
        ListNode p1 = head;
        while (head.next != null && k-- > 0) {
            p1 = p1.next;
        }
        if (k > 0) {
            return null;
        }
        ListNode p2 = head;
        while (p1 != null) {
            p1 = p1.next;
            p2 = p2.next;
        }
        return p2;
    }

    /**
     * 链表中环的入口结点
     */
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode fast = pHead, slow = pHead;
        while (fast != slow) {
            if (fast != null && fast.next != null) {
                fast = fast.next.next;
                slow = slow.next;
            }
        }
        fast = pHead;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     * 反转列表
     */
    public ListNode ReverseList(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode tmp = head.next;
        head.next = null;
        ListNode newHead = ReverseList(tmp);
        tmp.next = head;
        return newHead;
    }

    /**
     * 两个排序链表合并
     *
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        ListNode dummy = new ListNode(-1);
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                dummy.next = list1;
                list1 = list1.next;
            } else {
                dummy.next = list2;
                list2 = list2.next;
            }
        }
        while (list1 != null) {
            dummy.next = list1;
            list1 = list1.next;
        }
        while (list2 != null) {
            dummy.next = list2;
            list2 = list2.next;
        }
        return dummy.next;
    }

    /**
     * 递归写法
     */
    public ListNode MergeR(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        }
        if (list2 == null) {
            return list1;
        }
        if (list1.val < list2.val) {
            list1 = MergeR(list1.next, list2);
            return list1;
        } else {
            list1 = MergeR(list1, list2.next);
            return list2;
        }
    }

    /**
     * root2 是不是root1的子树
     */
    public boolean HasSubtree(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return false;
        }

        return isHasSubtree(root1, root2) || isHasSubtree(root1.left, root2) || isHasSubtree(root1.right, root2);

    }

    private boolean isHasSubtree(TreeNode root1, TreeNode root2) {
        if (root2 == null) {
            return true;
        }
        if (root1 == null) {
            return false;
        }
        if (root1.val != root2.val) {
            return false;
        }
        return isHasSubtree(root1.left, root2.left) && isHasSubtree(root1.right, root2.right);
    }

    /**
     * 二叉树的镜像变换
     */
    public void mirror(TreeNode root) {
        if (root == null) {
            return;
        }
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        mirror(root.left);
        mirror(root.right);
    }

    /**
     * 二叉树的左右子树是否对称
     * 也就是最左节点等于最优解点
     */
    public boolean isSym(TreeNode root) {
        if (root == null) {
            return false;
        }
        return isSym(root.left, root.right);
    }

    private boolean isSym(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val == right.val) {
            return true;
        }
        return isSym(left.left, right.right) && isSym(left.right, right.left);
    }

    public ArrayList<Integer> printMatrix(int[][] matrix) {
        ArrayList<Integer> res = new ArrayList<>(matrix.length * matrix[0].length);
        int r0 = 0, r1 = matrix.length - 1, c0 = 0, c1 = matrix[0].length - 1;
        while (r0 <= r1 && c0 <= c1) {
            for (int i = c0; i <= c1; i++) {
                res.add(matrix[r0][i]);//左起第一列
            }
            for (int i = r0 + 1; i <= r1; i++) {
                res.add(matrix[i][c1]);//从左到右最后一行
            }
            if (c0 != c1) {
                for (int i = c1 - 1; i >= c0; i--) {
                    res.add(matrix[r1][i]);
                }
            }
            if (r0 != r1) {
                for (int i = r1 - 1; i >= r0; i--) {
                    res.add(matrix[i][c0]);
                }
            }
            r0++;
            r1--;
            c0++;
            c1--;
        }
        return res;
    }

    /**
     * 实现最小栈
     */
    public static class MinStack {
        Stack<Integer> stack = new Stack<>();
        Stack<Integer> minStack = new Stack<>();

        public void push(int node) {
            stack.push(node);
            if (minStack.isEmpty()) {
                minStack.push(node);
            } else {
                minStack.push(Math.min(node, minStack.pop()));
            }
        }

        public int pop() {
            return stack.pop();
        }

        public int min() {
            return stack.peek();
        }
    }

    /**
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压 入栈的所有数字均不相等。
     * 例如序列 1,2,3,4,5 是某栈的压入顺序，序列 4,5,3,2,1 是该压栈序列对应的一个弹出序列，但 4,3,5,1,2 就不可能是该压栈序列的弹出序列。
     *
     * @param pushSequence
     * @param popSequence
     * @return
     */
    public boolean IsPopOrder(int[] pushSequence, int[] popSequence) {
        if (pushSequence.length != popSequence.length) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        int popIndex = 0;
        for (int i = 0; i < pushSequence.length; i++) {

            stack.push(pushSequence[i]);
            while (popIndex < pushSequence.length && !stack.isEmpty() && stack.peek() != popSequence[popIndex++]) {
                stack.pop();
            }
        }
        return stack.isEmpty();
    }

    /**
     * 从上往下打印二叉树
     * BFS 用层次遍历
     */
    public List<Integer> PrintFromTopToBottom(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> list = new ArrayList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            while (size-- > 0) {
                TreeNode tmp = queue.poll();
                if (tmp == null) {
                    continue;
                }
                list.add(tmp.val);
                if (root.left != null) {
                    queue.add(root.left);
                }
                if (root.right != null) {
                    queue.add(root.right);
                }
            }
        }
        return list;
    }

    public List<List<Integer>> PrintListFromTopToBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            while (size-- > 0) {
                TreeNode tmp = queue.poll();
                if (tmp == null) {
                    continue;
                }
                list.add(tmp.val);
                if (tmp.left != null) {
                    queue.add(tmp.left);
                }
                if (tmp.right != null) {
                    queue.add(tmp.right);
                }
            }
            if (list != null) {
                result.add(list);
            }

        }
        return result;
    }

    public List<List<Integer>> PrintZigzagListFromTopToBottom(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            int flag = 1;
            while (size-- > 0) {
                TreeNode tmp = queue.poll();
                if (tmp == null) {
                    continue;
                }
                list.add(tmp.val);
                if (tmp.left != null) {
                    queue.add(tmp.left);
                }
                if (tmp.right != null) {
                    queue.add(tmp.right);
                }
                flag *= -1;
            }
            if (list != null) {
                if (flag > 0) {
                    result.add(list);
                } else {
                    Collections.reverse(list);
                    result.add(list);
                }

            }

        }
        return result;
    }

    /**
     * 判断是不是二叉树的后续遍历
     *
     * @param sequence
     * @return
     */
    public boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null) {
            return false;
        }
        return backtrace(sequence, 0, sequence.length - 1);
    }

    private boolean backtrace(int[] seq, int l, int r) {
        if (l + 1 >= r) {
            return true;
        }//左右根 递归
        int rootVal = seq[r];
        int curIndex = l;
        while (curIndex < r && seq[curIndex] < seq[r]) {
            curIndex++;
        }
        for (int i = curIndex; i < r; i++) {
            if (seq[i] < rootVal) {
                return false;
            }
        }
        return backtrace(seq, l, curIndex - 1) && backtrace(seq, curIndex, r - 1);
    }

    private List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> minPathSum(TreeNode root, int target) {
        backtrace(root, target, new ArrayList<>());
        return res;
    }

    private void backtrace(TreeNode root, int target, List<Integer> list) {
        if (root == null) {
            return;
        }
        list.add(root.val);
        if (root.left == null && root.right == null) {
            if (target == root.val) {
                res.add(new ArrayList<>(list));
            }
        }
        backtrace(root.left, target - root.val, list);
        backtrace(root.right, target - root.val, list);
        list.remove(list.size() - 1);
    }

    /**
     * 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能 调整树中结点指针的指向
     */
    private TreeNode pre = null;
    private TreeNode pro = null;

    public TreeNode convert(TreeNode root) {
        inOrder(root);
        return pre;
    }

    private void inOrder(TreeNode root) {
        if (root == null) {
            return;
        }
        inOrder(root.left);
        root.left = pre;
        if (pre != null) {
            pre.right = root;
        }
        pre = root;
        if (pro == null) {
            pro = root;
        }
        inOrder(root.right);
    }

    //序列化二叉树
    private String tmpStr;

    public String Serialize(TreeNode root) {
        if (root == null) {
            return "#";
        }
        return Serialize(root.left) + " " + Serialize(root.right);
    }

    // 1 # 4 5 6 2
    public TreeNode DeSerialize(String str) {
        tmpStr = str;
        return DeSerialize();
    }

    private TreeNode DeSerialize() {
        if (tmpStr.length() == 0) {
            return null;
        }
        int index = tmpStr.indexOf(' ');
        String node = index == -1 ? tmpStr : tmpStr.substring(0, index);
        tmpStr = index == -1 ? "" : tmpStr.substring(index + 1, tmpStr.length() - 1);
        if (node == "#") {
            return null;
        }
        TreeNode t = new TreeNode(Integer.valueOf(node));
        t.left = DeSerialize();
        t.right = DeSerialize();
        return t;
    }

    /**
     * 1. 字符串的排列
     * 题目描述 输入一个字符串，按字典序打印出该字符串中字符的所有排列。例如输入字符串 abc，
     * 则打印出由字符 a, b, c 所能排列出来的所有字符串 abc, acb, bac, bca, cab 和 cba。
     */
    private List<String> permunate = new ArrayList<>();

    public List<String> permunation(String str) {
        if (str == null) {
            return permunate;
        }
        char[] ch = str.toCharArray();
        Arrays.sort(ch);
        backtrace(ch, new boolean[ch.length], new StringBuilder());
        return permunate;
    }

    //dfs 创建一个boolean 数组 判断这个有没有用过
    private void backtrace(char[] ch, boolean[] flag, StringBuilder sb) {
        if (sb.length() == ch.length) {
            permunate.add(sb.toString());
            return;
        }
        for (int i = 0; i < ch.length; i++) {
            if (flag[i]) {
                continue;
            }
            if (i != 0 && ch[i] == ch[i - 1] && !flag[i - 1]) {
                continue;
            }
            flag[i] = true;
            sb.append(ch[i]);
            backtrace(ch, flag, sb);
            sb.deleteCharAt(sb.length() - 1);
            flag[i] = false;
        }
    }

    /**
     * 数组中出现次数超过一半的数字
     * 投票法
     */
    public int MoreThanHalfNum(int[] nums) {
        if (nums == null) {
            return 0;
        }
        int major = nums[0];
        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            cnt = nums[i] == major ? cnt + 1 : cnt - 1;
            if (cnt == 0) {
                major = nums[i];
                cnt = 1;
            }
        }
        cnt = 0;
        for (int n : nums) {
            if (n == major) {
                cnt++;
            }
        }
        return cnt > nums.length / 2 ? major : 0;
    }

    /**
     * 最小的 K 个数
     * @param nums
     * @param k
     * @return
     */
    public ArrayList<Integer> GetLeastNumbers(int[] nums, int k) {
        PriorityQueue<Integer> queue=new PriorityQueue<>((o1, o2) -> (o2-o1));
        for(int i=0;i<nums.length;i++){
            queue.add(nums[i]);
            if(queue.size()>k){
                queue.poll();
            }
        }
        return new ArrayList<>(queue);
    }
    /**
     * 数据流中的中位数
     */
    static class MidInStream{
        //大顶堆
        PriorityQueue<Integer> left=new PriorityQueue<>((o1, o2) -> (o2-o1));
        //小顶堆
        PriorityQueue<Integer> right=new PriorityQueue<>();
        int N=0;
        public  void add(Integer val){
            if(N%2==0){
                left.add(val);
                right.add(left.poll());
            }else{
                right.add(val);
                left.add(right.poll());
            }
            N++;
        }
        public int getMedian(){
            if(N%2==0){
                return (left.peek()+right.peek())/2;
            }else{
                return right.peek();
            }
        }
    }

}