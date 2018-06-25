package Alogrithm.Facebook;

import java.util.*;


public class Facebook {

    static public class Interval {
        int start;
        int end;

        Interval() {
            start = 0;
            end = 0;
        }

        Interval(int s, int e) {
            start = s;
            end = e;
        }
    }

    //275 h-index
    public int hIndex(int[] citations) {
        int n = citations.length;
        int[] buckets = new int[n+1];
        for(int c : citations) {
            if(c >= n) {
                buckets[n]++;
            } else {
                buckets[c]++;
            }
        }
        int count = 0;
        for(int i = n; i >= 0; i--) {
            count += buckets[i];
            if(count >= i) {
                return i;
            }
        }
        return 0;
    }


    public int hIndex2(int[] citations) {
        int len = citations.length;
        int lo = 0, hi = len - 1;
        while (lo <= hi) {
            int med = (hi + lo) / 2;
            if (citations[med] == len - med) {
                return len - med;
            } else if (citations[med] < len - med) {
                lo = med + 1;
            } else {
                //(citations[med] > len-med), med qualified as a hIndex,
                // but we have to continue to search for a higher one.
                hi = med - 1;
            }
        }
        return len - lo;
    }

    static class TreeNode{
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    //    839. Merge Two Sorted Interval Lists lintcode
//    577. Merge K Sorted Interval Lists
//    two bst iterator
    class BSTITer{
        Stack<TreeNode> stack = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();
        BSTITer(TreeNode t1,TreeNode t2){
            TreeNode r1 = t1, r2 = t2;
            while(r1!=null){
                stack.add(r1);
                r1 = r1.left;
            }
            while(r2!=null){
                stack2.add(r2);
                r2 = r2.left;
            }
        }

        boolean hasNext(){
            return !stack.isEmpty() && !stack2.isEmpty();
        }

    }


    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> map=new HashMap<Character, Set<Character>>();
        Map<Character, Integer> degree=new HashMap<Character, Integer>();
        String result="";
        if(words==null || words.length==0) return result;
        for(String s: words){
            for(char c: s.toCharArray()){
                degree.put(c,0);
            }
        }
        for(int i=0; i<words.length-1; i++){
            String cur=words[i];
            String next=words[i+1];
            int length=Math.min(cur.length(), next.length());
            for(int j=0; j<length; j++){
                char c1=cur.charAt(j);
                char c2=next.charAt(j);
                if(c1!=c2){
                    Set<Character> set=new HashSet<Character>();
                    if(map.containsKey(c1)) set=map.get(c1);
                    if(!set.contains(c2)){
                        set.add(c2);
                        map.put(c1, set);
                        degree.put(c2, degree.get(c2)+1);
                    }
                    break;

                }
            }
        }
        Queue<Character> q=new LinkedList<Character>();
        for(char c: degree.keySet()){
            if(degree.get(c)==0) q.add(c);
        }
        while(!q.isEmpty()){
            char c=q.remove();
            result+=c;
            if(map.containsKey(c)){
                for(char c2: map.get(c)){
                    degree.put(c2,degree.get(c2)-1);
                    if(degree.get(c2)==0) q.add(c2);
                }
            }
        }
        if(result.length()!=degree.size()) return "";
        return result;
    }


    public String fractionAddition(String expression) {
        List < Character > sign = new ArrayList < > ();
        for (int i = 1; i < expression.length(); i++) {
            if (expression.charAt(i) == '+' || expression.charAt(i) == '-')
                sign.add(expression.charAt(i));
        }
        List < Integer > num = new ArrayList < > ();
        List < Integer > den = new ArrayList < > ();
        for (String sub: expression.split("\\+")) {
            for (String subsub: sub.split("-")) {
                if (subsub.length() > 0) {
                    String[] fraction = subsub.split("/");
                    num.add(Integer.parseInt(fraction[0]));
                    den.add(Integer.parseInt(fraction[1]));
                }
            }
        }
        if (expression.charAt(0) == '-')
            num.set(0, -num.get(0));
        int lcm = 1;
        for (int x: den) {
            lcm = lcm_(lcm, x);
        }

        int res = lcm / den.get(0) * num.get(0);
        for (int i = 1; i < num.size(); i++) {
            if (sign.get(i - 1) == '+')
                res += lcm / den.get(i) * num.get(i);
            else
                res -= lcm / den.get(i) * num.get(i);
        }
        int g = gcd(Math.abs(res), Math.abs(lcm));
        return (res / g) + "/" + (lcm / g);
    }
    public int lcm_(int a, int b) {
        return a * b / gcd(a, b);
    }
    public int gcd(int a, int b) {
        while (b != 0) {
            int t = b;
            b = a % b;
            a = t;
        }
        return a;
    }



    //    471. Encode String with Shortest Length
    public String encode(String s) {
        String[][] dp = new String[s.length()][s.length()];

        for(int l=0;l<s.length();l++) {
            for(int i=0;i<s.length()-l;i++) {
                int j = i+l;
                String substr = s.substring(i, j+1);
                // Checking if string length < 5. In that case, we know that encoding will not help.
                if(j - i < 4) {
                    dp[i][j] = substr;
                } else {
                    dp[i][j] = substr;
                    // Loop for trying all results that we get after dividing the strings into 2 and combine the   results of 2 substrings
                    for(int k = i; k<j;k++) {
                        if((dp[i][k] + dp[k+1][j]).length() < dp[i][j].length()){
                            dp[i][j] = dp[i][k] + dp[k+1][j];
                        }
                    }

                    // Loop for checking if string can itself found some pattern in it which could be repeated.
                    for(int k=0;k<substr.length();k++) {
                        String repeatStr = substr.substring(0, k+1);
                        if(repeatStr != null
                                && substr.length()%repeatStr.length() == 0
                                && substr.replaceAll(repeatStr, "").length() == 0) {
                            String ss = substr.length()/repeatStr.length() + "[" + dp[i][i+k] + "]";
                            if(ss.length() < dp[i][j].length()) {
                                dp[i][j] = ss;
                            }
                        }
                    }
                }
            }
        }

        return dp[0][s.length()-1];
    }



    //211
    static class WordDictionary {

        class Node{
            Node[] child = new Node[26];
            boolean isword;

        }
        /** Initialize your data structure here. */
        Node node;
        public WordDictionary() {
            node = new Node();
        }

        /** Adds a word into the data structure. */
        public void addWord(String word) {
            Node root = node;
            for(Character c:word.toCharArray()){
                if(root.child[c-'a'] == null)
                    root.child[c-'a'] = new Node();
                root = root.child[c-'a'];
            }
            root.isword = true;
        }

        /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
        public boolean search(String word) {
            return find(node,word);
        }
        private boolean find(Node root,String word){
            for(int i=0;i<word.length();i++){
                char c = word.charAt(i);
                if(c!='.'){
                    if(root.child[c-'a'] == null)
                        return false;
                    root = root.child[c-'a'];
                }
                else{
                    for(int j=0;j<26;j++){
                        if(root.child[j]!=null){
                            if(find(root.child[j],word.substring(i+1)))
                                return true;
                        }
                    }
                }
            }
            return root.isword;
        }
    }


    // 642
    static class AutocompleteSystem {
        class Pair{
            String s;
            int times;
            Pair(String s,int t){
                this.s = s;
                this.times = times;
            }
        }
        class Node{
            Map<Character,Node> child = new HashMap<>();
            PriorityQueue<Pair> pq = new PriorityQueue<>((a,b)->(a.times!=b.times?b.times-a.times:a.s.compareTo(b.s)));
        }

        Node root, cur;
        public AutocompleteSystem(String[] sentences, int[] times) {
            root = new Node();
            for(int i=0;i<sentences.length;i++){
                insert(root,sentences[i],times[i]);
            }

        }
        private void insert(Node root,String s,int t){
            Node curr = root;
            for(Character c:s.toCharArray()){
                if(!curr.child.containsKey(c)){
                    curr.child.put(c,new Node());
                }
                curr = curr.child.get(c);
                curr.pq.add(new Pair(s,t));
            }
        }

        public List<String> input(char c) {
            if(c == '#'){
                cur = root;
                return new LinkedList<>();
            }
            if(cur == null)
                cur = root;
            if(!cur.child.containsKey(c))
                return new LinkedList<>();
            cur = cur.child.get(c);
            List<String> res = new LinkedList<>();
            int k = 3;

            while(k-->0 && !cur.pq.isEmpty()){
                Pair p = cur.pq.poll();
                res.add(p.s);
            }
            return res;
        }
    }

    public static int maxSubArrayLen(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        map.put(0,-1);
        int sum = 0,max = 0;
        for(int i=0;i<nums.length;i++){
            sum += nums[i];
            if(map.containsKey(sum-k)){
                max = Math.max(max,i-map.get(sum-k));
            }
            else
                map.put(sum,i);
        }
        return max;
    }

    static class Pair{
        char c;
        int fre;
        Pair(char c,int f){
            this.c = c;
            this.fre = f;
        }
    }
    public static int leastInterval1(char[] tasks, int n) {
        Map<Character,Integer> map = new HashMap<>();
        for(char c:tasks){
            map.put(c,map.getOrDefault(c,0)+1);
        }
        PriorityQueue<Pair> pq = new PriorityQueue<>((a,b)->(b.fre-a.fre));
        for(char c:map.keySet()){
            pq.add(new Pair(c,map.get(c)));
        }
        StringBuilder sb = new StringBuilder();
        int count = 0;
        while(!pq.isEmpty() && sb.length()<tasks.length){
            Pair tmp = pq.poll();
            count++;
            sb.append(tmp.c);
            List<Pair> update = new LinkedList<>();
            for(int i=0;i<n;i--){
                if(sb.length()==tasks.length)
                    return count;
                if(pq.isEmpty()){
                    count++;
                    continue;
                }
                Pair in_slot = pq.poll();
                sb.append(in_slot.c);
                count++;
                in_slot.fre--;
                if(in_slot.fre>0)
                    update.add(in_slot);

            }
            for(Pair p:update)
                pq.add(p);
            tmp.fre--;
            if(tmp.fre>0)
                pq.add(tmp);
        }
        return count;
    }


    // two sum 2
    // assuming no duplicate in nums. from: 1point3acres
    public static List<int[]> threeSum(int[] nums, int target) {
        Arrays.sort(nums);

        List<int[]> ret = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            int left = i, right = nums.length - 1;
            while (left <= right) {
                int sum = nums[i] + nums[left] + nums[right] - target;
                if (sum == 0) {
                    if (i == left && left == right)
                        break;
                    ret.add(new int[]{nums[i], nums[left], nums[right]});
                    left++;
                    right--;
                } else if (sum > 0)
                    right--;
                else
                    left++;
            }
        }
        return ret;
    }



    // task schedular
    public static int leastInterval(char[] tasks, int n) {
        int[] map = new int[26];
        for (char c: tasks)
            map[c - 'A']++;
        Arrays.sort(map);
        int max_val = map[25] - 1, idle_slots = max_val * n;
        for (int i = 24; i >= 0 && map[i] > 0; i--) {
            idle_slots -= Math.min(map[i], max_val);
        }
        return idle_slots > 0 ? idle_slots + tasks.length : tasks.length;
    }

    public class DoublyListNode {
        int val;
        DoublyListNode next, prev;
        DoublyListNode(int val) {
            this.val = val;
            this.next = this.prev = null;
        }
    }

    //lintcode 378 bst to double linked list
    public DoublyListNode bstToDoublyList(TreeNode root) {
        // write your code here
        Stack<TreeNode> stack = new Stack<>();
        DoublyListNode prev = null, first = null;
        while(root!=null){
            stack.push(root);
            root = root.left;
        }
        while(!stack.isEmpty()){
            TreeNode node = stack.pop();
            DoublyListNode tmp = new DoublyListNode(node.val);
            if(prev == null){
                prev = tmp;
                first = prev;
            }
            else{
                prev.next = tmp;
                tmp.prev = prev;
                prev = tmp;
            }
            node = node.right;
            while(node!=null){
                stack.push(node);
                node = node.left;
            }

        }
        return first;
    }

    public static int[] kClosestNumbers(int[] A, int target, int k) {
        // write your code here
        int[] res = new int[k];
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>((a,b)->(Math.abs(b-target)-Math.abs(a-target)) !=0?Math.abs(b-target)-Math.abs(a-target):b-a );
        for(int i:A){
            pq.add(i);
            if(pq.size()>k)
                pq.poll();
        }
        int index = k-1;
        while(!pq.isEmpty()){
            res[index++] = pq.poll();
        }
        return res;

    }

    class Pairs{
        String s;
        int l;
        Pairs(String s,int l){
            this.s = s;
            this.l = l;
        }
    }
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> set = new HashSet<>();
        set.addAll(wordList);
        if(!set.contains(endWord))
            return 0;
        Queue<Pairs> queue = new LinkedList<>();
        queue.add(new Pairs(beginWord,1));
        while(!queue.isEmpty()){
            Pairs p = queue.poll();
            if(p.s.equals(endWord))
                return p.l;
            char[] tmp = p.s.toCharArray();
            for(int i=0;i<tmp.length;i++){
                char c = tmp[i];
                for(char x = 'a';x<='z';x++){
                    if(x == c)
                        continue;
                    tmp[i] = x;
                    String news = tmp.toString();
                    if(set.contains(news)){
                        queue.add(new Pairs(news,p.l+1));
                        set.remove(news);
                    }
                }
                tmp[i] = c;
            }
        }
        return 0;

    }

    static class NumMatrix {

        int[][] m;
        public NumMatrix(int[][] matrix) {
            int[][] m = new int[matrix.length][matrix[0].length+1];
            for(int i=0;i<matrix.length;i++){
                for(int j=1;j<matrix[0].length+1;j++){
                    m[i][j] = m[i][j-1]+matrix[i][j-1];
                }
            }
        }

        public int sumRegion(int row1, int col1, int row2, int col2) {
            int sum = 0;
            for(int i=row1;i<=row2;i++){
                sum += m[i][col2+1]-m[i][col1];
            }
            return sum;
        }
    }





    int ne=1;
    public void add(int ne){
        ne++;
    }

    public int largestArea(int[] nums){
        List<int[]> res = new LinkedList<>();
        int[] newH = new int[nums.length+1];
        for(int i=0;i<nums.length;i++){
            newH[i] = nums[i];
        }
        newH[newH.length-1] = -1;
        Stack<Integer> stack = new Stack<>();
        stack.add(-1);
        int max = 0;
        for(int i=0;i<nums.length;i++){
            while(stack.peek()!=-1 && newH[stack.peek()]>newH[i]){
                int H = stack.pop();
                //max = Math.max(max,newH[H]*(i-1-stack.peek()));
                int peek = stack.peek()+1;
                int area = newH[H]*(i-1-peek);
                if(max<area){
                    max = area;
                    res = new LinkedList<>();
                    res.add(new int[]{peek,i-1});
                }
                else{
                    res.add(new int[]{peek,i-1});
                }


            }
            stack.add(i);
        }
        for(int[] item:res){
            System.out.println(item[0]+","+item[1]);
        }
        return max;
    }

    class Point {
        int x;
        int y;
        Point() { x = 0; y = 0; }
        Point(int a, int b) { x = a; y = b; }
    }


    public Point[] kClosest(Point[] points, Point origin, int k) {
        // write your code here
        PriorityQueue<Point> pq = new PriorityQueue<Point>(new Comparator<Point>() {
            @Override
            public int compare(Point o1,Point o2){
                if(dist(o1,origin) == dist(o2,origin)){
                    if(o1.x == o2.x){
                        return o2.y-o1.y;
                    }
                    return o2.x-o1.x;
                }
                return dist(o2,origin)-dist(o1,origin);
            }

        });
        for(Point p:points){
            pq.add(p);
            if(pq.size()>k){
                pq.poll();
            }
        }
        Point[] res = new Point[k];
        int index = k;
        while(!pq.isEmpty()){
            res[index--] = pq.poll();
        }
        return res;
    }
    public int dist(Point a,Point b){
        return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y);
    }
//


    public List<Interval> mergeTwoInterval(List<Interval> list1, List<Interval> list2) {
        List<Interval> results = new ArrayList<>();
        if (list1 == null || list2 == null) {
            return results;
        }

        Interval last = null, curt = null;
        int i = 0, j = 0;
        while (i < list1.size() && j < list2.size()) {
            if (list1.get(i).start < list2.get(j).start) {
                curt = list1.get(i);
                i++;
            } else {
                curt = list2.get(j);
                j++;
            }

            last = merge(results, last, curt);
        }

        while (i < list1.size()) {
            last = merge(results, last, list1.get(i));
            i++;
        }

        while (j < list2.size()) {
            last = merge(results, last, list2.get(j));
            j++;
        }

        if (last != null) {
            results.add(last);
        }
        return results;
    }

    private Interval merge(List<Interval> results, Interval last, Interval curt) {
        if (last == null) {
            return curt;
        }

        if (curt.start > last.end) {
            results.add(last);
            return curt;
        }

        last.end = Math.max(last.end,curt.end);
        return last;
    }

    public List<Interval> mergeTwoIntervalwithOverlap(List<Interval> l1,List<Interval> l2){
        int i =0, j=0;
        List<Interval> res = new LinkedList<>();
        Interval curt = null, last = null;
        while(i<l1.size() && j<l2.size()){
            if(l1.get(i).start<l2.get(j).start){
                curt = l1.get(i++);
            }
            else{
                curt = l2.get(j++);}
            last = merge2(res,curt,last);
        }
        while(i<l1.size()){
            curt = l1.get(i++);
            last = merge2(res,curt,last);
        }
        while(j<l2.size()){
            curt = l2.get(j++);
            last = merge2(res,curt,last);
        }
        return res;
    }

    private Interval merge2(List<Interval> res,Interval curt,Interval last){
        if(last == null || last.end<=curt.start)
            return curt;
        res.add(new Interval(Math.max(last.start,curt.start),Math.min(last.end,curt.end)));
        return curt;

    }


    public String removeInvalidParentheses(String s) {
        String r = remove(s, new char[]{'(', ')'});
        String tmp = remove(new StringBuilder(r).reverse().toString(), new char[]{')', '('});
        return new StringBuilder(tmp).reverse().toString();
    }
    private String remove(String s, char[] p) {
        int stack = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == p[0])		stack++;
            if (s.charAt(i) == p[1])		stack--;
            if (stack < 0) {
                s = s.substring(0, i) + s.substring(i + 1);
                i--;
                stack = 0;
            }
        }
        return s;
    }

    public static int findKthLargest(int[] nums, int k) {
        return quickSelect(nums, 0, nums.length - 1, k);
    }
    private static int quickSelect(int[] nums, int left, int right, int k) {

        int pivot = nums[left];
        int i = left, j = right;
        while (i <= j) {
            while (i <= j && nums[i] > pivot) {
                i++;
            }
            while (i <= j && nums[j] < pivot) {
                j--;
            }
            if (i <= j) {
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                i++;
                j--;
            }
        }

        if (left + k - 1 <= j) {
            return quickSelect(nums, left, j, k);
        }
        if (left + k - 1 >= i) {
            return quickSelect(nums, i, right, k - (i - left));
        }
        return nums[j + 1];
    }

//    void merge(Leetcode.ListNode l1, Leetcode.ListNode l2) {
//        while (l1 != null) {
//            Leetcode.ListNode n1 = l1.next, n2 = l2.next;
//            l1.next = l2;
//
//            if (n1 == null)
//                break;
//
//            l2.next = n1;
//            l1 = n1;
//            l2 = n2;
//        }
//    }



    public List<List<Integer>> verticalOrder(TreeNode root) {
        Map<Integer,List<Integer>> map = new HashMap<>();
        Queue<Integer> queue = new LinkedList<>();
        Queue<TreeNode> queue2 = new LinkedList<>();
        List<List<Integer>> res = new LinkedList<>();
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        if(root == null)
            return res;
        queue.add(0);
        queue2.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue2.poll();
            int level = queue.poll();
            if(!map.containsKey(level))
                map.put(level,new LinkedList<>());
            map.get(level).add(node.val);
            if(node.left!=null){
                queue.add(level-1);
                queue2.add(node.left);
            }
            if(node.right!=null){
                queue.add(level-1);
                queue2.add(node.right);
            }
            min = Math.min(min,level);
            max = Math.min(max,level);

        }
        for(int i=min;i<=max;i++){
            if(map.containsKey(i)){
                res.add(map.get(i));
            }
        }
        return res;
    }

    public static class RandomizedCollection {
        ArrayList<Integer> nums;
        HashMap<Integer, Set<Integer>> locs;
        java.util.Random rand = new java.util.Random();

        /**
         * Initialize your data structure here.
         */
        public RandomizedCollection() {
            nums = new ArrayList<Integer>();
            locs = new HashMap<Integer, Set<Integer>>();
        }

        /**
         * Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
         */
        public boolean insert(int val) {
            boolean contain = locs.containsKey(val);
            if (!contain) locs.put(val, new LinkedHashSet<Integer>());
            locs.get(val).add(nums.size());
            nums.add(val);
            return !contain;
        }

        /**
         * Removes a value from the collection. Returns true if the collection contained the specified element.
         */
        public boolean remove(int val) {
            boolean contain = locs.containsKey(val);
            if (!contain) return false;
            int loc = locs.get(val).iterator().next();
            locs.get(val).remove(loc);
            if (loc < nums.size() - 1) {
                int lastone = nums.get(nums.size() - 1);
                nums.set(loc, lastone);
                locs.get(lastone).remove(nums.size() - 1);
                locs.get(lastone).add(loc);
            }
            nums.remove(nums.size() - 1);

            if (locs.get(val).isEmpty()) locs.remove(val);
            return true;
        }

        /**
         * Get a random element from the collection.
         */
        public int getRandom() {
            return nums.get(rand.nextInt(nums.size()));
        }
    }

    //Merge k sorted arrays
    public static int[] mergeK(int[][] arrays){
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>((a,b)->(a[0]-b[0]));
        int size = 0;
        for(int i=0;i<arrays.length;i++){
            size += arrays[i].length;
            pq.add(new int[]{arrays[i][0],i,0});
        }
        int index = 0;
        int[] ans = new int[size];
        while(!pq.isEmpty()){
            int[] tmp = pq.poll();
            ans[index++] = tmp[0];
            tmp[2]++;
            if(tmp[2]<arrays[tmp[1]].length){
                tmp[0] = arrays[tmp[1]][tmp[2]];
                pq.add(tmp);
            };
        }
        return ans;
    }

    //332
    public List<String> findItinerary(String[][] tickets) {
        Map<String, PriorityQueue<String>> targets = new HashMap<>();
        for (String[] ticket : tickets)
            targets.computeIfAbsent(ticket[0], k -> new PriorityQueue()).add(ticket[1]);
        List<String> route = new LinkedList();
        Stack<String> stack = new Stack<>();
        stack.push("JFK");
        while (!stack.empty()) {
            while (targets.containsKey(stack.peek()) && !targets.get(stack.peek()).isEmpty())
                stack.push(targets.get(stack.peek()).poll());
            route.add(0, stack.pop());
        }

        try{
            System.out.println();
        }
        catch (Exception e){
            e.printStackTrace();
        }

        return route;
    }



    public static int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("."), v2 = version2.split(".");
        int length = Math.max(v1.length,v2.length);
        Integer n1 = null, n2 = null;

        for(int i=0;i<length;i++){
            if(i<v1.length){
                n1 = Integer.parseInt(v1[i]);
            }
            else{
                n1 = 0;
            }
            if(i<v2.length){
                n2 = Integer.parseInt(v2[i]);
            }
            else{
                n2 = 0;
            }

            if(n1<n2)
                return -1;
            else if(n1>n2)
                return 1;
            else
                continue;


        }
        return 0;

    }


    public int trap(int[] height) {
        Stack<Integer> stack = new Stack<>();
        int count = 0;
        for(int i=0;i<height.length;i++){
            if(!stack.isEmpty() && height[stack.peek()]<height[i]){
                int tmp = stack.pop();
                if(!stack.isEmpty()){
                    count += (Math.min(height[stack.peek()],height[i])-height[tmp])*(i-stack.peek()-1);
                }
            }
            stack.push(i);
        }
        return count;
    }

    public String countAndSay(int n) {
        String res = "1";
        for(int i=0;i<n-1;i++){
            int index = 0;
            String tmp = "";
            while(index<res.length()){
                int j = index;
                while(j<res.length() && res.charAt(index) == res.charAt(j))
                    j++;
                tmp += (j-index)+(char)res.charAt(index);
                index = j;
            }
            res = tmp;
        }
        return res;
    }


    List<String> res = new LinkedList<>();
    public List<String> restoreIpAddresses(String s) {
        dfs(s,"",0);
        return res;
    }
    private void dfs(String s, String tmp, int index){
        if(index == s.length()){
            res.add(tmp.substring(0,tmp.length()-1));
            return;
        }
        for(int i=index;i<index+3;i++){
            String substr = s.substring(index,i+1);
            if(isValid(substr)){
                substr = substr+".";
                dfs(s,tmp+"."+substr,i+1);
            }
        }
    }
    private boolean isValid(String s){
        if(s.charAt(0) == '0')
            return s.equals("0");
        int i = Integer.parseInt(s);
        return 0<=i && i<=255;
    }




    //282
    public List<String> addOperator(String s, int tmp){
        List<String> res = new LinkedList<>();
        dfs(s,0,0,tmp,res,"");
        return res;
    }
    private void dfs(String s,int pos,int sum,int tmp,List<String> res,String t){
        if(pos == s.length()){
            if(sum == tmp){
                res.add(t);
            }
            return;
        }
        for(int i=pos;i<s.length();i++){
            if(i!=pos && s.charAt(pos) == '0')
                break;
            int cur = Integer.parseInt(s.substring(pos,i+1));
            if(pos == 0)
                dfs(s,i+1,sum+cur,tmp,res,t+cur);
            else{
                dfs(s,i+1,sum+cur,tmp,res,t+"+"+cur);
                dfs(s,i+1,sum-cur,tmp,res,t+"-"+cur);
            }
        }

    }


    public static List<Integer> subset(int[] nums){
        List<Integer> res = new LinkedList<>();
        res.add(1);
        for(int i:nums){
            int size = res.size();
            for(int j=0;j<size;j++){
                res.add(i*res.get(j));
            }

        }
        return res.subList(1,res.size());
    }






    public static void main(String[] args){
        Facebook facebook = new Facebook();
        //System.out.println(facebook.ladderLength("aab","bbb",new LinkedList<String>(){{add("bab");add("bbb");}}));

//        A: [1,5], [10,14], [16,18]
//        B: [2,6], [8,10], [11,20]
//      Interval i1 = new Interval(1,5);
//      Interval i2 = new Interval(10,14);
//      Interval i3 = new Interval(16,18);
//      Interval i4 = new Interval(2,6);
//      Interval i5 = new Interval(8,10);
//      Interval i6 = new Interval(11,20);
//      List<Interval> l1 = new LinkedList<>(Arrays.asList(i1,i2,i3));
//      List<Interval> l2 = new LinkedList<>(Arrays.asList(i4,i5,i6));
//      facebook.mergeTwoIntervalwithOverlap(l1,l2);

        System.out.println(subset(new int[]{2,3,4}));













    }

}
