package Alogrithm.Google;

import java.util.*;

public class Google {
    public static boolean canTransform(String start, String end) {
        if(!start.replace("X","").equals(end.replace("X","")))
            return false;
        int i = 0, j = 0;
        while(i<start.length() && j<end.length()){
            while(i<start.length() && start.charAt(i)=='X')i++;
            while(j<end.length() && end.charAt(j)=='X')j++;

            if(i == start.length() && j == end.length())
                return true;
            if(i == start.length() || j == end.length())
                return false;
            if(start.charAt(i)!=end.charAt(j))
                return false;
            if(start.charAt(i) == 'L' && i<j)
                return false;
            if(start.charAt(i) == 'R' && i>j)
                return false;
            i++;
            j++;
        }
        return i==start.length() && j == end.length();
    }
    static class RandomListNode {
        int label;
        RandomListNode next, random;
        RandomListNode(int x) { this.label = x; }
    }


    public static RandomListNode copyRandomList(RandomListNode head) {
        if(head == null) return head;
        RandomListNode root = new RandomListNode(head.label);
        RandomListNode newHead = root;
        while(head != null){
            if(head.random != null){
                root.random = new RandomListNode(head.random.label);
            }else root.random = null;

            if(head.next != null){
                root.next = new RandomListNode(head.next.label);
            }else root.next = null;
            root = root.next;
            head = head.next;
        }
        return newHead;
    }

    public static int seat_index(int[] nums){
        TreeSet<Integer> set = new TreeSet<>();
        set.add(-1);
        int index = -1,gap = 0;
        for(int i=0;i<nums.length+1;i++){
            if(i == nums.length || nums[i] == 1){
                Integer left = set.floor(i);
                if (left!=null && i-left-1>0){
                    if(gap<i-left-1){
                        gap = i-left-1;
                        index = left+(i-left)/2;
                    }
                }
                set.add(i);
            }
        }


        return index;
    }




    public int[] findRedundantDirectedConnection(int[][] edges) {
        int[] can1 = {-1, -1};
        int[] can2 = {-1, -1};
        int[] parent = new int[edges.length + 1];
        for (int i = 0; i < edges.length; i++) {
            if (parent[edges[i][1]] == 0) {
                parent[edges[i][1]] = edges[i][0];
            } else {
                can2 = new int[] {edges[i][0], edges[i][1]};
                can1 = new int[] {parent[edges[i][1]], edges[i][1]};
                edges[i][1] = 0;
            }
        }
        for (int i = 0; i < edges.length; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < edges.length; i++) {
            if (edges[i][1] == 0) {
                continue;
            }
            int child = edges[i][1], father = edges[i][0];
            if (root(parent, father) == child) {
                if (can1[0] == -1) {
                    return edges[i];
                }
                return can1;
            }
            parent[child] = father;
        }
        return can2;
    }

    int root(int[] parent, int i) {
        while (i != parent[i]) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }

    public static String compress(String str){
        // a(b(c){2}){2}d
        Stack<String> stack = new Stack<>();
        String res = "";
        int num = 0,index = 0;
        while(index<str.length()){
            char c = str.charAt(index);
            if(c == '('){
                stack.push(res);
                res = "";
                num = 0;
            }
            else if(c == ')'){
                if(index<str.length()-1 && str.charAt(index+1) == '{'){
                    int j = index+2;
                    while(j<str.length() && Character.isDigit(str.charAt(j))){
                        num = num*10+str.charAt(j)-'0';
                        j++;
                    }
                    index = j;
                }
                else
                    num = 1;
                String tmp = "";
                for(int k=0;k<num;k++)
                    tmp += res;
                res = stack.pop()+tmp;
                num = 0;
            }
            else if(Character.isAlphabetic(c))
                res += c;
            index++;

        }
        return res;
    }


    //313
    public static int nthSuperUglyNumber1(int n, int[] primes) {
        int[] ugly = new int[n];
        int[] idx = new int[primes.length];
        int[] val = new int[primes.length];
        Arrays.fill(val, 1);

        int next = 1;
        for (int i = 0; i < n; i++) {
            ugly[i] = next;

            next = Integer.MAX_VALUE;
            for (int j = 0; j < primes.length; j++) {
                //skip duplicate and avoid extra multiplication
                if (val[j] == ugly[i]) val[j] = ugly[idx[j]++] * primes[j];
                //find next ugly number
                next = Math.min(next, val[j]);
            }
        }

        return ugly[n - 1];
    }
    public int nthSuperUglyNumberHeap(int n, int[] primes) {
        int[] ugly = new int[n];

        PriorityQueue<Num> pq = new PriorityQueue<>();
        for (int i = 0; i < primes.length; i++) pq.add(new Num(primes[i], 1, primes[i]));
        ugly[0] = 1;

        for (int i = 1; i < n; i++) {
            ugly[i] = pq.peek().val;
            while (pq.peek().val == ugly[i]) {
                Num nxt = pq.poll();
                pq.add(new Num(nxt.p * ugly[nxt.idx], nxt.idx + 1, nxt.p));
            }
        }
        System.out.println(pq.size());

        return ugly[n - 1];
    }

    private class Num implements Comparable<Num> {
        int val;
        int idx;
        int p;

        public Num(int val, int idx, int p) {
            this.val = val;
            this.idx = idx;
            this.p = p;
        }

        @Override
        public int compareTo(Num that) {
            return this.val - that.val;
        }
    }

    //410
    public static int splitArray(int[] nums, int m) {
        long l = 0;
        long r = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            r += nums[i];
            if (l < nums[i]) {
                l = nums[i];
            }
        }
        long ans = r;
        while (l <= r) {
            long mid = (l + r) >> 1;
            long sum = 0;
            int cnt = 1;
            for (int i = 0; i < n; i++) {
                if (sum + nums[i] > mid) {
                    cnt ++;
                    sum = nums[i];
                } else {
                    sum += nums[i];
                }
            }
            if (cnt <= m) {
                ans = Math.min(ans, mid);
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return (int)ans;
    }


    public int numDistinctIslands(int[][] grid) {
        Set<String> set = new HashSet<>();

        for(int i=0;i<grid.length;i++){
            for(int j=0;j<grid[0].length;j++){
                if(grid[i][j] == 1){
                    StringBuilder s = new StringBuilder();
                    dfs(i,j,s,grid,"X");
                    if(s.length()!=0)
                        set.add(s.toString());

                }
            }
        }
        return set.size();

    }
    private void dfs(int i,int j,StringBuilder s,int[][] grid,String pos){
        if(i<0 || j<0 || i>=grid.length || j>=grid[0].length || grid[i][j]!=1)
            return;
        grid[i][j] = 0;
        s.append(pos);
        dfs(i+1,j,s,grid,"D");
        dfs(i-1,j,s,grid,"U");
        dfs(i,j+1,s,grid,"R");
        dfs(i,j-1,s,grid,"L");


    }
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int row = matrix.length;
        int col = matrix[0].length;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < row; i ++) {
            int[] colSum = new int[col];
            for (int j = i; j < row; j ++) {
                for (int c = 0; c < col; c ++) {
                    colSum[c] += matrix[j][c];
                }
                max = Math.max(max, findMax(colSum, k));
            }
        }
        return max;
    }

    private int findMax(int[] nums, int k) {
        int max = Integer.MIN_VALUE;
        int sum = 0;
        TreeSet<Integer> s = new TreeSet();
        s.add(0);

        for(int i = 0;i < nums.length; i ++){
            int t = sum + nums[i];
            sum = t;
            Integer gap = s.ceiling(sum - k);
            if(gap != null) max = Math.max(max, sum - gap);
            s.add(t);
        }

        return max;
    }


    //363
    public int maxSumSubmatrix1(int[][] matrix, int k) {
        int row = matrix.length, col = matrix[0].length;
        int[][] area = new int[row][col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                area[i][j] = matrix[i][j];
                if(i>=1)
                    area[i][j] += matrix[i-1][j];
                if(j>=1)
                    area[i][j] += matrix[i][j-1];
                if(i>=1 && j>=1)
                    area[i][j] -= matrix[i-1][j-1];
            }
        }
        int max = Integer.MIN_VALUE;
        for(int r1 = 0;r1<row;r1++){
            for(int c1 = 0;c1<col;c1++){
                for(int r2 = r1;r2<row;r2++){
                    for(int c2 = c1;c2<col;c2++){
                        int cur_area = area[r2][c2];
                        if(c1>=1)
                            cur_area -= area[r2][c1-1];
                        if(r1>=1)
                            cur_area -= area[r1-1][c2];
                        if(c1>=1 && r1>=1)
                            cur_area += area[r1-1][c1-1];
                        if(cur_area<=k){
                            max = Math.max(max,cur_area);
                        }
                    }
                }
            }
        }
        return max;
    }

    //s1: aBc   s2 axx#bb#cc#c    #b-> delete previous one #c-> to upper
    public static boolean isToStringSame(String s1,String s2){
        boolean upper = false;
        int i = 0,j = 0;
        while(j<s2.length()){
            if(s2.charAt(j) == '#'){
                if(j< s2.length() && s2.charAt(j+1) == 'c')
                    upper = !upper;
                j = j+2;
            }
            else
                j++;
        }
        i = s1.length()-1;
        j = s2.length()-1;
        int count = 0;
        while(i>=0 && j>=0){
            char c = s2.charAt(j);
            if(j>0 && s2.charAt(j) == 'c' && s2.charAt(j-1) == '#') {
                upper = !upper;
                j -= 2;
            }
            else if(j>0 && s2.charAt(j) == 'b' && s2.charAt(j-1) == '#'){
                count += 1;
                j -= 2;
            }
            else if(count>0){
                count--;
                j--;
            }
            else{
                if(upper)
                    c = Character.toUpperCase(c);
                if(c != s1.charAt(i))
                    return false;
                else {
                    i--;
                    j--;
                }
            }

        }
        return true;

    }


    public static int quick_select(int[] nums,int start,int end,int k){
        if(start == end)
            return nums[start];
        int i = start,j = end;
        int pivot = nums[i];
        while(i < j){
            while(i<j && nums[j]>=pivot)
                j--;
            nums[i] = nums[j];
            while(i<j && nums[i]<= pivot)
                i++;
            nums[j] = nums[i];

        }
        nums[i] = pivot;
        if(start<i && k<i)
            return quick_select(nums,start,i-1,k);
        else if(i<end && k>i)
            return quick_select(nums,i+1,end,k);
        return nums[i];

    }
    public int findKthLargest(int[] nums, int k) {
        int res = helper(nums,0,nums.length-1,nums.length-k);
        return res;
    }
    private int helper(int[] nums,int start,int end,int k){
        if(start == end)
            return nums[start];
        int i = start, j = end, pi = nums[start];
        while(i<j){
            while(i<j && nums[j]>=pi)
                j--;
            nums[i] = nums[j];
            while(i<j && nums[i]<=pi)
                i++;
            nums[j] = nums[i];
        }
        nums[i] = pi;
        if(start<i && i>k)
            return helper(nums,start,i-1,k);
        else if(i<end && i<k)
            return helper(nums,i+1,end,k);

        return nums[k];

    }


    /*
    Quack<T> a special data sturcature that can pop a item randomly from head or tail
    The element should insert in sorted order  eg [1,3,4,5,6,7]
    And call pop() for the first time may return 1 or 7
    Implement a O(n) method that call pop() -> get a sorted list
     */
    public static class Quack<T>{
        ArrayDeque<T> queue = new ArrayDeque<>();


        Quack(T[] t){
            queue = new ArrayDeque<>();
            for(T tt:t)
                queue.add(tt);
        }

        Quack(List<T> t){
            queue.addAll(t);
        }

        public T pop(){
            Random r = new Random();
            T res = null;
            int i = r.nextInt(2);
            if(i == 0)
                res = queue.pollFirst();
            else if(i == 1)
                res = queue.pollLast();
            return res;
        }
        public boolean isEmpty(){
            return queue.isEmpty();
        }
    }



//    public static <T extends Comparable<T>> List<T> returnOrderList(Quack<T> quack){
//
//        List<T> list = new ArrayList<>();
//        ArrayDeque<T> larger = new ArrayDeque<T>();
//        ArrayDeque<T> smaller = new ArrayDeque<T>();
//
//        T prev = null, next = null;
//        while(!quack.isEmpty()){
//            if(next == null)
//                prev = quack.pop();
//            else
//                prev = next;
//            if(!quack.isEmpty())
//                next = quack.pop();
//            else
//                next = null;
//            if(next == null || prev.compareTo(next)>0){
//                larger.addFirst(prev);
//
//            }
//            else if(prev.compareTo(next) > 0)
//                smaller.add(prev);
//            else if(prev.compareTo(next) == 0){
//                smaller.add(prev);
//                smaller.add(next);
//                next = null;
//            }
//
//
//        }
//        if(next!=null)
//            larger.addFirst(next);
//        for(T t:smaller)
//            list.add(t);
//        for(T t:larger)
//            list.add(t);
//
//        return list;
//    }




    public static int smallestDistancePair(int[] nums, int k) {
        Arrays.sort(nums);
        int len = nums.length;
        int max = nums[len-1]-nums[0];
        int min = max;
        for(int i=1;i<len;i++){
            min = Math.min(min,nums[i]-nums[i-1]);
        }

        while(min<max){
            int cnt = 0;
            int mid = min+(max-min)/2;
            for(int i=0,j=0;i<len;i++){
                while(j<len&&(nums[j]-nums[i])<=mid)j++;
                cnt += j-i-1;
            }
            if(cnt<k)
                min = mid+1;
            else
                max = mid;




        }
        return min;

    }

    class TimeStampMap<Key,time,Value>{

    }

//    767. Reorganize String




    public static void main(String[] args){



        Quack<Integer> quack = new Quack<Integer>(new Integer[]{2,2,2,3,3,4,4,5,5,6});
        smallestDistancePair(new int[]{1,3,6,9},4);
        StringBuilder sb = new StringBuilder();




    }
}

