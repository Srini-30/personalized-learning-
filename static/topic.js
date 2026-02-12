document.getElementById("topicForm")
.addEventListener("submit", async function(e){

e.preventDefault();

const topic = document.getElementById("topicInput").value;
const level = document.getElementById("level").value;

const res = await fetch("/api/topic",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({topic, level})
});

const data = await res.json();

console.log(data);


// ===== RENDER =====

document.getElementById("summary").innerText = data.summary;


// bullets
const bullets = document.getElementById("bullets");
bullets.innerHTML="";
data.bullets.forEach(b=>{
    bullets.innerHTML += `<li>${b}</li>`;
});


// definitions
const defs = document.getElementById("definitions");
defs.innerHTML="";
data.definitions.forEach(d=>{
    defs.innerHTML += `<li>${d}</li>`;
});


// examples
const ex = document.getElementById("examples");

if(data.examples){
    ex.innerHTML = data.examples.join("<br><br>");
}
else{
    ex.innerText = data.example;
}


// ✅ SAVE QUIZ
sessionStorage.setItem("quiz",
    JSON.stringify(data.quiz || data.quick_quiz)
);

sessionStorage.setItem("topic", topic);


// SHOW QUIZ BUTTON
document.getElementById("startQuiz").style.display="block";

});


// QUIZ NAV
document.getElementById("startQuiz")
.addEventListener("click",()=>{

window.location="/quiz";

});
