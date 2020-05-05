package Data::MachineLearning::Elements::NaiveBayes;

use Lingua::EN::Tokenizer::Offsets qw(get_tokens);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my $v = $ml->tokenize('Little Mary had a little lamb.'); # [a, had, lamb, little, mary]

  $ml->train(@messages);

  my $y = $ml->nb_predict('little lamb');

=head1 ATTRIBUTES

=head2 k

Smoothing factor

=cut

has k => (
    is      => 'ro',
    default => sub { 0.5 },
);

=head2 tokens

=cut

has tokens => (
    is      => 'rw',
    default => sub { {} },
);

=head2 token_spam_counts

=cut

has token_spam_counts => (
    is      => 'rw',
    default => sub { {} },
);

=head2 token_ham_counts

=cut

has token_ham_counts => (
    is      => 'rw',
    default => sub { {} },
);

=head2 spam_messages

=cut

has spam_messages => (
    is      => 'rw',
    default => sub { 0 },
);

=head2 ham_messages

=cut

has ham_messages => (
    is      => 'rw',
    default => sub { 0 },
);

=head1 METHODS

=head2 tokenize

  $v = $ml->tokenize($string);

=cut

sub tokenize {
    my ($self, $string) = @_;
    my $words = get_tokens($string);
    my %seen;
    $seen{lc $_}++
        for map { s/[[:punct:]]//gr } @$words;
    return [ sort keys %seen ];
}

=head2 train

  $ml->train(@messages);

Where the B<messages> are a list of hash references,

  [ { text => $string, is_spam => $bool }, { ... } ]

=cut

sub train {
    my ($self, @messages) = @_;
    for my $message (@messages) {
        if ($message->{is_spam}) {
            $self->spam_messages( $self->spam_messages + 1 );
        }
        else {
            $self->ham_messages( $self->ham_messages + 1 );
        }
        for my $token (@{ $self->tokenize($message->{text}) }) {
            $self->tokens->{$token}++;
            if ($message->{is_spam}) {
                $self->token_spam_counts->{$token}++;
            }
            else {
                $self->token_ham_counts->{$token}++;
            }
        }
    }
}

sub _probabilities {
    my ($self, $token) = @_;
    my $spam = $self->token_spam_counts->{$token} || 0;
    my $ham = $self->token_ham_counts->{$token} || 0;
    my $p_token_spam = ($spam + $self->k) / ($self->spam_messages + 2 * $self->k);
    my $p_token_ham = ($ham + $self->k) / ($self->ham_messages + 2 * $self->k);
    return $p_token_spam, $p_token_ham;
}

=head2 nb_predict

  $y = $ml->nb_predict($text);

=cut

sub nb_predict {
    my ($self, $text) = @_;
    my %text_tokens;
    @text_tokens{ @{ $self->tokenize($text) } } = undef;
    my $log_prob_if_spam = 0;
    my $log_prob_if_ham = 0;
    for my $token (sort keys %{ $self->tokens }) {
        my ($prob_if_spam, $prob_if_ham) = $self->_probabilities($token);
        if (exists $text_tokens{$token}) {
            $log_prob_if_spam += log $prob_if_spam;
            $log_prob_if_ham += log $prob_if_ham;
        }
        else {
            $log_prob_if_spam += log(1 - $prob_if_spam);
            $log_prob_if_ham += log(1 - $prob_if_ham);
        }
    }
    my $prob_if_spam = exp $log_prob_if_spam;
    my $prob_if_ham = exp $log_prob_if_ham;
    return $prob_if_spam / ($prob_if_spam + $prob_if_ham);
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/009-naive-bayes.t>

L<Lingua::EN::Tokenizer::Offsets>

L<Moo::Role>

=cut
