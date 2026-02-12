const quiz = JSON.parse(sessionStorage.getItem("quiz"));
const topic = sessionStorage.getItem("topic");

if(!quiz){
    document.body.innerHTML="<h2>No quiz found. Learn first.</h2>";
}

const box = document.getElementById("quizBox");

quiz.forEach((q,i)=>{

box.innerHTML += `
<div>

<p><b>${i+1}. ${q.q}</b></p>

<input type="radio" name="q${i}" value="${q.a}"> ${q.a}
<br>

<input type="radio" name="q${i}" value="wrong"> Wrong

</div>
`;

});


document.getElementById("submitQuiz")
.onclick = async function(){

let score=0;

quiz.forEach((q,i)=>{

const ans = document.querySelector(`input[name="q${i}"]:checked`);

if(ans && ans.value===q.a){
    score++;
}

});

await fetch("/api/save_quiz_score",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({
topic,
score,
total:quiz.length
})
});

alert(`Score: ${score}/${quiz.length}`);

window.location="/";

}
