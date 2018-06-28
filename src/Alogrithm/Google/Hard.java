package Alogrithm.Google;


import java.util.Arrays;
import java.util.PriorityQueue;

public class Hard {
    //857

        public double mincostToHireWorkers(int[] quality, int[] wage, int K) {
            int N = quality.length;
            Worker[] workers = new Worker[N];
            for (int i = 0; i < N; ++i)
                workers[i] = new Worker(quality[i], wage[i]);
            Arrays.sort(workers);

            double ans = 1e9;
            int sumq = 0;
            PriorityQueue<Integer> pool = new PriorityQueue();
            for (Worker worker: workers) {
                pool.offer(-worker.quality);
                sumq += worker.quality;
                if (pool.size() > K)
                    sumq += pool.poll();
                if (pool.size() == K)
                    ans = Math.min(ans, sumq * worker.ratio());
            }

            return ans;
        }

    class Worker implements Comparable<Worker> {
        public int quality, wage;
        public Worker(int q, int w) {
            quality = q;
            wage = w;
        }

        public double ratio() {
            return (double) wage / quality;
        }

        public int compareTo(Worker other) {
            return Double.compare(ratio(), other.ratio());
        }
        }


        // 132
    public int minCut(String s) {
        int[] cut = new int[s.length()];
        char[] c = s.toCharArray();
        boolean[][] dp = new boolean[s.length()][s.length()];

        for(int i=0;i<s.length();i++){
            int min = i;    // given a string, the maxium cut is the number of current length-1  abc-> a b c
            for(int j=i;j>=0;j--){
                if(c[j]==c[i] && (j+1>i-1 || dp[i-1][j+1])){
                    min = Math.min(min,j==0?0:cut[j-1]+1);
                    dp[i][j] = true;
                    }
            }
            cut[i] = min;
        }
        return cut[s.length()-1];
        }


    public static void main(String[] args){

            Hard h = new Hard();
            h.minCut("aab");

    }


    }


