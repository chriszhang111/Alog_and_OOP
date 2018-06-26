package Alogrithm.OOD.Card;

import java.util.ArrayList;
import java.util.Random;


public class Deck <E extends Card> {
    private ArrayList<E> cards;
    private int dealtIndex = 0;

    public Deck() {
    }

    public void setDeckOfCards(ArrayList<E> deckOfCards) {
        cards = deckOfCards;
    }

    public void shuffle() {
        Random random = new Random();
        for (int i = 0; i < cards.size(); i++) {
            int j = random.nextInt(i);
            E card1 = cards.get(i);
            E card2 = cards.get(j);
            cards.set(i, card2);
            cards.set(j, card1);
        }
    }


}
