package Alogrithm.Facebook;

import java.util.*;

//leetcode 381
class RandomizedCollection {

    /** Initialize your data structure here. */
    List<Integer> list = new ArrayList<>();
    Map<Integer,Set<Integer>> map = new HashMap<>();
    Random rand = new Random();
    public RandomizedCollection() {

    }

    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        boolean flag = true;
        if(map.containsKey(val))
            flag = false;
        else{
            map.put(val,new HashSet<>());

        }
        map.get(val).add(list.size());
        list.add(val);
        return flag;
    }

    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val))
            return false;
        if(list.get(list.size()-1)!=val){
            int index = map.get(val).iterator().next();
            int last = list.get(list.size()-1);
            map.get(last).remove(list.size()-1);
            map.get(last).add(index);
            map.get(val).remove(index);
            map.get(val).add(list.size()-1);

            list.set(index,last);
        }
        map.get(val).remove(list.size()-1);
        if(map.get(val).size()==0)
            map.remove(val);
        list.remove(list.size()-1);
        return true;
    }

    /** Get a random element from the collection. */
    public int getRandom() {
        int index = rand.nextInt(list.size());
        return list.get(index);
    }
    public static void main(String[] args){
        RandomizedCollection r = new RandomizedCollection();
        r.insert(0);
        r.remove(0);
        r.insert(-1);
        r.remove(0);
        System.out.println(r.getRandom());
    }
}
