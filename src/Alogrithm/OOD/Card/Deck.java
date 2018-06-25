package Alogrithm.OOD.Card;

import java.util.ArrayList;
import java.util.Random;


public class Deck <T extends Card> {
    private ArrayList<T> cards;
    private int dealtIndex = 0;

    public Deck() {
    }

    public void setDeckOfCards(ArrayList<T> deckOfCards) {
        cards = deckOfCards;
    }

    public void shuffle() {
        Random random = new Random();
        for (int i = 0; i < cards.size(); i++) {
            int j = random.nextInt(i);
            T card1 = cards.get(i);
            T card2 = cards.get(j);
            cards.set(i, card2);
            cards.set(j, card1);
        }
    }


}
