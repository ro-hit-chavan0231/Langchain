from langchain_text_splitters import CharacterTextSplitter

text = """
Football is one of the most popular sports in the world. Millions of people watch and play football every day. It is loved because it is exciting, simple to understand, and brings people together. Football is played in almost every country, from small villages to large cities. Children, teenagers, and adults all enjoy this wonderful game.

A football match is played between two teams. Each team has eleven players. The main aim of the game is to score more goals than the other team. Players use their feet to pass, control, and kick the ball. Only the goalkeeper is allowed to use their hands, and only inside the penalty area. A standard football match lasts for 90 minutes, divided into two halves of 45 minutes each.

Football is not only about scoring goals. It also teaches many important life skills. Players learn teamwork, discipline, patience, and respect for others. They must communicate well with their teammates and work together to win the game. Football also helps people stay healthy because it improves fitness, strength, speed, and stamina.

Many famous football tournaments are watched around the world. The FIFA World Cup is the biggest football competition and is held every four years. Clubs also compete in exciting leagues and tournaments such as the Premier League and the UEFA Champions League.

Some of the greatest football players have inspired millions of fans. Lionel Messi is known for his excellent dribbling and passing skills. Cristiano Ronaldo is famous for his powerful shots, speed, and goal-scoring ability. Their dedication and hard work encourage young players to follow their dreams.

Football is more than just a game. It creates friendships, builds confidence, and brings people from different cultures together. Whether people play on a professional field or in a local park, football provides happiness and excitement. It teaches the importance of fair play, determination, and never giving up. For these reasons, football continues to be one of the most loved and respected sports in the world.
"""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator =""
)

result = splitter.split_text(text)

print(result)